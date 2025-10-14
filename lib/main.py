#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading, queue
import numpy as np, cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import settings as S
from pipeline import create_pipeline_and_sink, build_pipeline
from tcn_classifier import TCNOnnxClassifier

# ─────────────────────────────────────────────────────────────────────────────
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
         (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]

def _f(obj, name):
    attr = getattr(obj, name, None)
    return float(attr() if callable(attr) else attr)

def _maybe(obj, name):
    a = getattr(obj, name, None)
    if a is None: return None
    try: return a() if callable(a) else a
    except Exception: return None

# ─────────────────────────────────────────────────────────────────────────────
# Hailo 메타 → 사람 bbox/랜드마크 픽셀 좌표(+confidence)로 변환
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
                c = _maybe(p, "confidence")
                if c is None: c = _maybe(p, "score")
                if c is None: c = _maybe(p, "visibility")
                if c is None: c = 0.5
                pts.append((px, py, float(c)))
            people.append({
                "bbox": [int(xmin*w), int(ymin*h), int((xmin+bw)*w), int((ymin+bh)*h)],
                "kpt": pts
            })
    except Exception:
        pass
    return people

# ─────────────────────────────────────────────────────────────────────────────
def draw_overlay(frame, people, show_kpt_index=False, conf_thr=0.65):
    """고신뢰 키포인트/엣지(>= conf_thr)만 그리기 + 기하학적 이상선 제거.
       또한, 코(0)↔어깨(5,6) 엣지는 제외."""
    if not people: return
    H, W = frame.shape[:2]
    max_len2 = (max(W, H) * 0.6) ** 2  # 너무 긴 엣지는 스킵

    for person in people:
        x1,y1,x2,y2 = map(int, person.get("bbox", [0,0,0,0]))
        pts = person.get("kpt", [])

        # 박스 색상
        hi = [p for p in pts if len(p) >= 3 and float(p[2]) >= conf_thr]
        color_box = (0,255,0) if len(hi) >= 6 else (160,160,160)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color_box, 2)

        # 점
        vis = [False]*len(pts)
        for j, pt in enumerate(pts):
            if len(pt) < 2: continue
            x, y = int(pt[0]), int(pt[1])
            c = float(pt[2]) if len(pt) >= 3 else 1.0
            if c < conf_thr:  # 저신뢰 무시
                continue
            vis[j] = True
            cv2.circle(frame, (x,y), 3, (255,255,255), -1)
            if show_kpt_index:
                cv2.putText(frame, str(j), (x+3, y-3), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(frame, str(j), (x+3, y-3), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,0), 1, cv2.LINE_AA)

        # 엣지 (코-어깨 제거, 양 끝 고신뢰, 과도 길이 제거)
        for a,b in EDGES:
            if (a==0 and b in (5,6)) or (b==0 and a in (5,6)):
                continue  # 코-어깨 라인 제거
            if a < len(pts) and b < len(pts) and vis[a] and vis[b]:
                x1_, y1_ = int(pts[a][0]), int(pts[a][1])
                x2_, y2_ = int(pts[b][0]), int(pts[b][1])
                dx, dy = x1_ - x2_, y1_ - y2_
                if (dx*dx + dy*dy) <= max_len2:
                    cv2.line(frame, (x1_, y1_), (x2_, y2_), (0,200,255), 2)

# ─────────────────────────────────────────────────────────────────────────────
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
def on_new_sample_data(sink, ctx):
    if ctx["stop"].is_set():
        return Gst.FlowReturn.EOS
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
        try: q.get_nowait()
        except queue.Empty: pass
        try: q.put_nowait(pkt)
        except queue.Full: pass
    return Gst.FlowReturn.OK

# ─────────────────────────────────────────────────────────────────────────────
def app_worker(ctx):
    data_q, stop, appsrc = ctx["data_q"], ctx["stop"], ctx["appsrc"]
    clf: TCNOnnxClassifier = ctx.get("clf", None)
    show_idx = getattr(S, "SHOW_KPT_INDEX", False)
    stride = int(getattr(S, "TCN_STRIDE", 1))
    conf_thr = float(getattr(S, "MIN_KPT_CONF", 0.65))
    if conf_thr > 1.5:  # 퍼센트로 들어오면 0~1로 환산
        conf_thr = conf_thr / 100.0
    only_best = True  # 최상 신뢰도 1인만 그림/분류
    frame_i = 0
    last_txt = "[TCN] warming…"
    last_conf_txt = ""

    def iou(a,b):
        if not a or not b: return 0.0
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
        inter = iw*ih
        areaA = max(0,ax2-ax1)*max(0,ay2-ay1)
        areaB = max(0,bx2-bx1)*max(0,by2-by1)
        uni = areaA + areaB - inter if inter>0 else areaA + areaB
        return inter/uni if uni>0 else 0.0

    def kpt_conf_mean(p):
        k = p.get("kpt", [])
        vals = [float(pt[2]) for pt in k if len(pt) >= 3]
        return (float(np.mean(vals)) if vals else 0.0), len(vals)

    prev_bbox = None

    while not stop.is_set():
        try:
            pkt = data_q.get(timeout=0.5)
        except queue.Empty:
            continue

        # 선택(신뢰도 평균 최대 1명)
        sel = TCNOnnxClassifier._select_person(pkt["people"]) if pkt["people"] else None

        # conf_mean 임계치 미만이면 "no person" 처리
        display_people = []
        cur_bbox = None
        if sel is not None:
            cmean, n = kpt_conf_mean(sel)
            if cmean >= conf_thr:
                display_people = [sel] if only_best else (pkt["people"] or [])
                cur_bbox = sel["bbox"]
                last_conf_txt = f"[KPT] conf_mean={cmean:.2f} (n={n})"
            else:
                last_conf_txt = f"[KPT] low conf ({cmean:.2f})"

        # 오버레이(저신뢰면 그리지 않음)
        frame = pkt["frame"]
        draw_overlay(frame, display_people, show_kpt_index=show_idx, conf_thr=conf_thr)

        # 주체 변경 시 버퍼 리셋
        if clf is not None and prev_bbox is not None and cur_bbox is not None:
            if iou(prev_bbox, cur_bbox) < 0.3:
                clf.reset()
        prev_bbox = cur_bbox

        # stride 반영해 update 호출 (저신뢰면 빈 리스트 -> no person)
        if clf is not None and (frame_i % stride == 0):
            pred = clf.update(display_people, pkt["size"])
            if pred is None:
                last_txt = "[TCN] no person" if not display_people else "[TCN] warming…"
            else:
                last_txt = f"[TCN] {pred['label']} ({pred['score']:.2f})"

        # 좌상단 상태 텍스트(분류 + 신뢰도)
        cv2.putText(frame, last_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, last_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
        if last_conf_txt:
            cv2.putText(frame, last_conf_txt, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, last_conf_txt, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)

        try:
            buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
            buf.fill(0, frame.tobytes())
            appsrc.emit("push-buffer", buf)
        except Exception:
            break

        frame_i += 1

# ─────────────────────────────────────────────────────────────────────────────
def _attach_bus_watch(loop, pipe, name, stop_evt):
    bus = pipe.get_bus()
    def on_msg(bus, msg):
        t = msg.type
        if t in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
            try:
                if t == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print(f"[{name}] ERROR:", err, dbg)
                else:
                    print(f"[{name}] EOS")
            finally:
                stop_evt.set(); loop.quit()
        return True
    bus.add_signal_watch()
    bus.connect("message", on_msg)

def main():
    print(build_pipeline(S)); print("==============")

    # ❶ 추론 파이프라인 (appsink=data_sink)
    pipe, sink = create_pipeline_and_sink(S)
    try:
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
    except Exception:
        pass

    # ❷ 표시 파이프라인 (appsrc→autovideosink)
    disp_pipe, appsrc = create_display_pipeline(S.SRC_WIDTH, S.SRC_HEIGHT)

    # ❸ ONNX TCN 분류기 초기화 (경로는 settings.py 또는 아래 기본값)
    onnx_path = getattr(S, "TCN_ONNX", "models/251013_10_44_40/tcn.onnx")
    json_path = getattr(S, "TCN_JSON", "models/251013_10_44_40/tcn.json")
    try:
        clf = TCNOnnxClassifier(onnx_path=onnx_path, json_path=json_path)
        if not clf.ok:
            print("[TCN] classifier unavailable:", clf.err)
            clf = None
        else:
            print("[TCN] ready:", "NCT" if clf.channels_first else "NTC",
                  "| C=", clf.expected_C, "| T=", clf.expected_T,
                  "| features=", clf.features, "| norm=", clf.norm,
                  "| classes=", clf.classes)
    except Exception as e:
        print("[TCN] init failed:", e); clf = None

    loop = GLib.MainLoop()
    stop_evt = threading.Event()
    ctx = {"stop": stop_evt, "data_q": queue.Queue(maxsize=2), "appsrc": appsrc, "clf": clf}

    # 콜백 연결
    sink.connect("new-sample", on_new_sample_data, ctx)

    # 버스 워치
    _attach_bus_watch(loop, pipe, "infer", stop_evt)
    _attach_bus_watch(loop, disp_pipe, "display", stop_evt)

    # 파이프라인 시작(실패 시 종료)
    ret1 = pipe.set_state(Gst.State.PLAYING)
    ret2 = disp_pipe.set_state(Gst.State.PLAYING)
    if ret1 == Gst.StateChangeReturn.FAILURE:
        print("[infer] pipeline start failed"); return
    if ret2 == Gst.StateChangeReturn.FAILURE:
        print("[display] pipeline start failed"); return

    # 응용 워커 시작
    t_app = threading.Thread(target=app_worker, args=(ctx,), daemon=True)
    t_app.start()

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        try: appsrc.emit("end-of-stream")
        except Exception: pass
        pipe.set_state(Gst.State.NULL)
        disp_pipe.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()