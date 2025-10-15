# core/hailo_pose_stream.py
from __future__ import annotations
import os, threading, queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from . import settings as S
from .tcn_classifier import TCNOnnxClassifier

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU 강제

# ─────────────────────────────────────────────────────────────────────────────
def _f(obj, name):
    attr = getattr(obj, name, None)
    return float(attr() if callable(attr) else attr)

def _maybe(obj, name):
    a = getattr(obj, name, None)
    if a is None: return None
    try: return a() if callable(a) else a
    except Exception: return None

def extract_people_from_buf(buf, w, h) -> List[Dict[str, Any]]:
    """Hailo meta → [{'bbox':[x1,y1,x2,y2], 'kpt':[(x,y,c), ...]}]"""
    people: List[Dict[str, Any]] = []
    try:
        import hailo
        roi = hailo.get_roi_from_buffer(buf)
        if not roi: return people
        dets = roi.get_objects_typed(hailo.HAILO_DETECTION)
        for d in dets:
            lms = d.get_objects_typed(hailo.HAILO_LANDMARKS)
            if not lms: continue
            b = d.get_bbox()
            xmin, ymin, bw, bh = _f(b,"xmin"), _f(b,"ymin"), _f(b,"width"), _f(b,"height")
            pts = []
            for p in lms[0].get_points():
                px = int((p.x()*bw + xmin) * w)
                py = int((p.y()*bh + ymin) * h)
                c = _maybe(p, "confidence") or _maybe(p, "score") or _maybe(p, "visibility") or 0.5
                pts.append((px, py, float(c)))
            people.append({
                "bbox": [int(xmin*w), int(ymin*h), int((xmin+bw)*w), int((ymin+bh)*h)],
                "kpt": pts
            })
    except Exception:
        pass
    return people

def _conf_mean(person: Dict[str, Any]) -> float:
    k = person.get("kpt", [])
    vals = [float(pt[2]) for pt in k if len(pt) >= 3]
    return float(np.mean(vals)) if vals else 0.0

def _bbox_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def _select_best(people: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not people: return None
    scored = [ (_conf_mean(p), _bbox_area(p["bbox"]), i, p) for i,p in enumerate(people) ]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][3]

def _iou(a, b) -> float:
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

# ─────────────────────────────────────────────────────────────────────────────
class HailoPoseStream:
    """Start → read → stop. Returns RGB frame, people, optional TCN result."""
    def __init__(
        self,
        conf_thr: float = 0.65,
        stride: int = 1,
        iou_reset_th: float = 0.30,
        onnx_path: Optional[str] = None,
        json_path: Optional[str] = None,
        prefer_cpu: bool = True,
    ):
        self.conf_thr = conf_thr if conf_thr <= 1.5 else conf_thr/100.0
        self.stride = max(1, int(stride))
        self.iou_reset_th = float(iou_reset_th)

        self._stop = threading.Event()
        self._loop: Optional[GLib.MainLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pipe = None
        self._appsink = None
        self._worker: Optional[threading.Thread] = None
        self._data_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2)
        self._out_q: "queue.Queue[Tuple[Optional[np.ndarray], List[Dict[str, Any]], Optional[Dict[str, Any]], Tuple[int,int]]]" = queue.Queue(maxsize=1)
        self._frame_i = 0

        # classifier (optional)
        self._clf: Optional[TCNOnnxClassifier] = None
        self._prev_bbox = None
        if onnx_path is None:
            onnx_path = getattr(S, "TCN_ONNX", None)
        if json_path is None:
            json_path = getattr(S, "TCN_JSON", None)
        if onnx_path and json_path:
            try:
                self._clf = TCNOnnxClassifier(onnx_path=onnx_path, json_path=json_path, prefer_cpu=prefer_cpu)
                if not self._clf.ok:
                    print("[TCN] unavailable:", self._clf.err)
                    self._clf = None
                else:
                    print("[TCN] ready:", "NCT" if self._clf.channels_first else "NTC",
                          "| C=", self._clf.expected_C, "| T=", self._clf.expected_T,
                          "| features=", self._clf.features, "| norm=", self._clf.norm,
                          "| classes=", self._clf.classes)
            except Exception as e:
                print("[TCN] init failed:", e); self._clf=None

    # ── pipeline builder ─────────────────────────────────────────────────────
    @staticmethod
    def _make_desc(io_mode: int) -> str:
        CAM = S.CAM
        SRC_WIDTH, SRC_HEIGHT, SRC_FPS = S.SRC_WIDTH, S.SRC_HEIGHT, S.SRC_FPS
        CROPPER_SO = S.CROPPER_SO
        HEF = S.HEF
        POST_SO = S.POST_SO
        POST_FUNC = S.POST_FUNC
        return f"""
v4l2src device={CAM} io-mode={io_mode} do-timestamp=true !
image/jpeg, width={SRC_WIDTH}, height={SRC_HEIGHT}, framerate={SRC_FPS}/1 !
jpegdec !
videoconvert ! videoscale !
video/x-raw,format=RGB,width={SRC_WIDTH},height={SRC_HEIGHT},pixel-aspect-ratio=1/1 !
queue name=inference_wrapper_input_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailocropper name=crop so-path={CROPPER_SO} function-name=create_crops use-letterbox=true resize-method=inter-area internal-offset=true
hailoaggregator name=agg
crop. ! queue name=bypass_q leaky=no max-size-buffers=20 max-size-bytes=0 max-size-time=0 ! agg.sink_0
crop. ! queue name=inf_scale_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! videoscale n-threads=2 qos=false ! videoconvert n-threads=2 !
video/x-raw !
queue name=inf_hnet_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailonet name=hnet hef-path={HEF} batch-size=2 vdevice-group-id=1 force-writable=true !
queue name=inf_post_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 !
hailofilter name=post so-path={POST_SO} function-name={POST_FUNC} qos=false !
queue name=inf_out_q leaky=no max-size-buffers=3 max-size-bytes=0 max-size-time=0 ! agg.sink_1
agg. ! queue leaky=no max-size-buffers=3 max-size-bytes=0 !
appsink name=data_sink emit-signals=true sync=false max-buffers=1 drop=true
"""

    # ── public API ───────────────────────────────────────────────────────────
    def start(self):
        Gst.init(None)

        # io-mode 2 → 실패 시 0으로 재시도
        last_exc = None
        for io_mode in (2, 0):
            desc = self._make_desc(io_mode)
            try:
                self._pipe = Gst.parse_launch(desc)
                break
            except Exception as e:
                last_exc = e
                self._pipe = None
        if self._pipe is None:
            raise RuntimeError(f"Failed to build pipeline: {last_exc}")

        self._appsink = self._pipe.get_by_name("data_sink")
        if self._appsink is None:
            raise RuntimeError("appsink 'data_sink' not found")

        # appsink 안전설정
        try:
            self._appsink.set_property("max-buffers", 1)
            self._appsink.set_property("drop", True)
            self._appsink.set_property("emit-signals", True)
        except Exception:
            pass

        self._appsink.connect("new-sample", self._on_new_sample, None)

        # GLib 루프 + 버스 워치
        self._loop = GLib.MainLoop()
        bus = self._pipe.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # 파이프라인 시작
        ret = self._pipe.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._pipe.set_state(Gst.State.NULL)
            raise RuntimeError("pipeline start failed")

        # 스레드 시작
        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()
        self._worker = threading.Thread(target=self._worker_fn, daemon=True)
        self._worker.start()

    def read(self, timeout: Optional[float] = 0.0):
        """→ (frame_rgb, people, cls_result, (w,h))"""
        try:
            item = self._out_q.get(timeout=timeout) if (timeout and timeout>0) else self._out_q.get_nowait()
            return item
        except queue.Empty:
            return None, [], None, (S.SRC_WIDTH, S.SRC_HEIGHT)

    def stop(self):
        self._stop.set()
        try:
            if self._appsink is not None:
                self._appsink.set_property("emit-signals", False)
        except Exception:
            pass
        if self._pipe is not None:
            self._pipe.set_state(Gst.State.NULL)
        if self._loop is not None:
            try: self._loop.quit()
            except Exception: pass

    # ── internals ────────────────────────────────────────────────────────────
    def _on_bus_message(self, bus, msg):
        t = msg.type
        if t in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
            try:
                if t == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print("[GST] ERROR:", err, dbg)
                else:
                    print("[GST] EOS")
            finally:
                self._stop.set()
                try: 
                    if self._loop: self._loop.quit()
                except Exception:
                    pass
        return True

    def _on_new_sample(self, sink, _ud):
        if self._stop.is_set():
            return Gst.FlowReturn.EOS

        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s0 = caps.get_structure(0)
        w  = int(s0.get_value('width'))
        h  = int(s0.get_value('height'))
        fmt = (s0.get_value('format') or 'RGB') if s0.has_field('format') else 'RGB'
        fmt = str(fmt).upper()

        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            arr = np.frombuffer(mi.data, np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mi)

        # appsink 캡스 포맷에 따라 RGB 보정
        if fmt == 'BGR':
            frame_rgb = arr[:, :, ::-1]
        else:
            frame_rgb = arr

        people = extract_people_from_buf(buf, w, h)
        pkt = {"size": (w,h), "people": people, "frame": frame_rgb}
        try:
            self._data_q.put_nowait(pkt)
        except queue.Full:
            try: _ = self._data_q.get_nowait()
            except queue.Empty: pass
            try: self._data_q.put_nowait(pkt)
            except queue.Full: pass
        return Gst.FlowReturn.OK

    def _push_latest(self, frame_rgb, people, cls_result, size):
        try:
            while True:
                self._out_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._out_q.put_nowait((frame_rgb, people, cls_result, size))
        except queue.Full:
            pass

    def _worker_fn(self):
        while not self._stop.is_set():
            try:
                pkt = self._data_q.get(timeout=0.5)
            except queue.Empty:
                continue

            frame_rgb = pkt["frame"]
            w, h = pkt["size"]
            people = pkt["people"]
            sel = _select_best(people) if people else None

            cls_result = None
            if self._clf is not None and sel is not None:
                cmean = _conf_mean(sel)
                cur_bbox = sel.get("bbox")
                if cmean >= self.conf_thr:
                    if self._prev_bbox is not None and cur_bbox is not None:
                        if _iou(self._prev_bbox, cur_bbox) < self.iou_reset_th:
                            self._clf.reset()
                    self._prev_bbox = cur_bbox
                    if (self._frame_i % self.stride) == 0:
                        pred = self._clf.update([sel], (w,h))  # only selected
                        if pred is not None:
                            cls_result = dict(pred)
                            cls_result["conf_mean"] = float(cmean)
                            cls_result["bbox"] = list(cur_bbox) if cur_bbox is not None else None
                else:
                    self._clf.reset()
                    self._prev_bbox = None

            self._frame_i += 1
            self._push_latest(frame_rgb, people, cls_result, (w,h))

# ── singleton helpers ────────────────────────────────────────────────────────
_stream_singleton: Optional[HailoPoseStream] = None

def start_stream(**kwargs) -> HailoPoseStream:
    global _stream_singleton
    if _stream_singleton is not None:
        return _stream_singleton
    _stream_singleton = HailoPoseStream(**kwargs)
    _stream_singleton.start()
    return _stream_singleton

def read_latest(timeout: Optional[float] = 0.0):
    return _stream_singleton.read(timeout) if _stream_singleton else (None, [], None, (S.SRC_WIDTH, S.SRC_HEIGHT))

def stop_stream():
    global _stream_singleton
    if _stream_singleton is not None:
        _stream_singleton.stop()
        _stream_singleton = None
