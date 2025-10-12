#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# Qt/HighGUI 안 씀. 굳이 설정할 필요 없음.
# os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time, threading, queue
import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import settings as S
from pipeline import create_pipeline_and_sink, build_pipeline

def extract_people_from_buf(buf, w, h):
    people=[]
    try:
        import hailo
        roi = hailo.get_roi_from_buffer(buf)
        if not roi: return people
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for d in dets:
            lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms: continue
            b = d.get_bbox()
            xmin = float(b.xmin() if callable(getattr(b,"xmin",None)) else b.xmin)
            ymin = float(b.ymin() if callable(getattr(b,"ymin",None)) else b.ymin)
            bw   = float(b.width() if callable(getattr(b,"width",None)) else b.width)
            bh   = float(b.height() if callable(getattr(b,"height",None)) else b.height)
            pts=[]
            for p in lms[0].get_points():
                px = (p.x()*bw + xmin) * w
                py = (p.y()*bh + ymin) * h
                pts.append((px, py))
            people.append({"bbox":[xmin*w, ymin*h, (xmin+bw)*w, (ymin+bh)*h], "kpt":pts})
    except Exception:
        pass
    return people

# appsink 콜백: 데이터만 큐로 푸시
def on_new_sample_data(sink, ctx):
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK
    buf = sample.get_buffer(); caps = sample.get_caps()
    s0 = caps.get_structure(0)
    w = s0.get_value('width'); h = s0.get_value('height')

    ok, mi = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK
    try:
        arr = np.frombuffer(mi.data, dtype=np.uint8)
        frame = arr.reshape((h, w, 3)).copy()  # BGR
    finally:
        buf.unmap(mi)

    people = extract_people_from_buf(buf, w, h)
    pts_ns = int(buf.pts) if buf.pts not in (None, Gst.CLOCK_TIME_NONE) else None
    pkt = {"pts_ns": pts_ns, "size": (w,h), "people": people, "frame": frame}

    q = ctx["data_q"]
    try:
        q.put_nowait(pkt)
    except queue.Full:
        try: q.get_nowait()
        except queue.Empty: pass
        try: q.put_nowait(pkt)
        except queue.Full: pass
    return Gst.FlowReturn.OK

# 응용 워커: 블로킹 대기 → 처리
def app_worker(ctx):
    q = ctx["data_q"]
    while not ctx["stop"].is_set():
        try:
            pkt = q.get(timeout=0.5)
        except queue.Empty:
            continue
        # 여기서 응용 로직(네 분석/전송/저장) 수행
        print(f"[APP] pts={pkt['pts_ns']} people={len(pkt['people'])} size={pkt['size']}")

def main():
    print("=== PIPE ==="); print(build_pipeline(S)); print("==============")

    pipe, data_sink = create_pipeline_and_sink(S)
    loop = GLib.MainLoop()
    ctx = {"loop": loop, "stop": threading.Event(), "data_q": queue.Queue(maxsize=2)}

    data_sink.connect("new-sample", on_new_sample_data, ctx)

    # 응용 워커 시작
    t = threading.Thread(target=app_worker, args=(ctx,), daemon=True)
    t.start()

    # GLib 메인루프는 메인 스레드에서
    try:
        pipe.set_state(Gst.State.PLAYING)
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        ctx["stop"].set()
        pipe.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
