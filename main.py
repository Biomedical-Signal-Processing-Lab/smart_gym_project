#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, threading, queue
import numpy as np, cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import settings as S
from pipeline import create_pipeline_and_sink, build_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 스켈레톤 에지(COCO-17)
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
         (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]

def _f(obj, name):
    """Hailo bbox 필드가 메서드/속성 어느 쪽이든 float로 얻기"""
    attr = getattr(obj, name, None)
    return float(attr() if callable(attr) else attr)

# ─────────────────────────────────────────────────────────────────────────────
# Hailo 메타 → 사람 bbox/랜드마크 픽셀 좌표로 변환
def extract_people_from_buf(buf, w, h):
    people = []
    try:
        import hailo
        roi = hailo.get_roi_from_buffer(buf)
        if not roi:
            return people
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for d in dets:
            lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms:
                continue
            b = d.get_bbox()
            xmin, ymin, bw, bh = _f(b,"xmin"), _f(b,"ymin"), _f(b,"width"), _f(b,"height")

            pts = []
            for p in lms[0].get_points():
                px = int((p.x()*bw + xmin) * w)
                py = int((p.y()*bh + ymin) * h)
                pts.append((px, py))
            people.append({
                "bbox": [int(xmin*w), int(ymin*h), int((xmin+bw)*w), int((ymin+bh)*h)],
                "kpt": pts
            })
    except Exception:
        pass
    return people

# ─────────────────────────────────────────────────────────────────────────────
# OpenCV로 bbox/키포인트/스켈레톤 그리기
def draw_overlay(frame, people):
    for person in people:
        x1,y1,x2,y2 = person["bbox"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        pts = person["kpt"]
        for (x,y) in pts:
            cv2.circle(frame, (x,y), 3, (255,255,255), -1)
        for a,b in EDGES:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0,200,255), 2)

# ─────────────────────────────────────────────────────────────────────────────
# 표시용 파이프라인(appsrc → autovideosink)
def create_display_pipeline(width, height):
    Gst.init(None)
    desc = """
appsrc name=disp_src is-live=true block=true format=time do-timestamp=true !
videoconvert !
autovideosink name=preview sync=false
"""
    pipe = Gst.parse_launch(desc)
    appsrc = pipe.get_by_name("disp_src")
    caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={width},height={height}")
    appsrc.set_property("caps", caps)
    return pipe, appsrc

# ─────────────────────────────────────────────────────────────────────────────
# appsink 콜백: 프레임+BGR 및 메타를 패킷으로 만들어 큐로 푸시
def on_new_sample_data(sink, ctx):
    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    caps = sample.get_caps()
    s0 = caps.get_structure(0)
    w  = s0.get_value('width')
    h  = s0.get_value('height')

    ok, mi = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.OK
    try:
        frame = np.frombuffer(mi.data, np.uint8).reshape((h, w, 3)).copy()  # BGR
    finally:
        buf.unmap(mi)

    people = extract_people_from_buf(buf, w, h)
    pts_ns = int(buf.pts) if buf.pts not in (None, Gst.CLOCK_TIME_NONE) else None

    pkt = {"pts_ns": pts_ns, "size": (w,h), "people": people, "frame": frame}

    q = ctx["data_q"]
    try:
        q.put_nowait(pkt)
    except queue.Full:
        try: q.get_nowait()  # 오래된 1개 드롭
        except queue.Empty: pass
        try: q.put_nowait(pkt)
        except queue.Full: pass

    return Gst.FlowReturn.OK

# ─────────────────────────────────────────────────────────────────────────────
# 응용 워커: OpenCV로 그린 뒤 appsrc로 푸시(표시)
def app_worker(ctx):
    data_q, stop, appsrc = ctx["data_q"], ctx["stop"], ctx["appsrc"]
    while not stop.is_set():
        try:
            pkt = data_q.get(timeout=0.5)
        except queue.Empty:
            continue

        frame = pkt["frame"]
        draw_overlay(frame, pkt["people"])

        # appsrc로 밀어넣기 (BGR 그대로)
        buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
        buf.fill(0, frame.tobytes())
        # do-timestamp=true 이므로 PTS/DURATION 없어도 자동 타임스탬프 부여
        appsrc.emit("push-buffer", buf)

# ─────────────────────────────────────────────────────────────────────────────
def _attach_bus_watch(loop, pipe, name):
    bus = pipe.get_bus()
    def on_msg(bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[{name}] ERROR:", err, dbg)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print(f"[{name}] EOS")
            loop.quit()
        return True
    bus.add_signal_watch()
    bus.connect("message", on_msg)

def main():
    print(build_pipeline(S)); print("==============")

    # ❶ 추론 파이프라인 (appsink=data_sink)
    pipe, sink = create_pipeline_and_sink(S)

    # ❷ 표시 파이프라인 (appsrc→autovideosink)
    disp_pipe, appsrc = create_display_pipeline(S.SRC_WIDTH, S.SRC_HEIGHT)

    loop = GLib.MainLoop()
    stop_evt = threading.Event()
    ctx = {
        "stop":   stop_evt,
        "data_q": queue.Queue(maxsize=2),
        "appsrc": appsrc,
    }

    # 콜백 연결
    sink.connect("new-sample", on_new_sample_data, ctx)

    # 버스 워치(에러/EOS 시 종료)
    _attach_bus_watch(loop, pipe,     "infer")
    _attach_bus_watch(loop, disp_pipe,"display")

    # 파이프라인 시작
    pipe.set_state(Gst.State.PLAYING)
    disp_pipe.set_state(Gst.State.PLAYING)

    # 응용 워커 시작
    t_app = threading.Thread(target=app_worker, args=(ctx,), daemon=True)
    t_app.start()

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        # appsrc EOS(선택)
        try:
            appsrc.emit("end-of-stream")
        except Exception:
            pass
        pipe.set_state(Gst.State.NULL)
        disp_pipe.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
