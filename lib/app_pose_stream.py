# app_pose_stream.py
# -*- coding: utf-8 -*-
"""
Minimal, reusable stream library that returns (frame, people, cls_result).
- No drawing, no preview window.
- ONNX runs on CPU.
- Selects a single person (highest mean keypoint confidence; tie → larger bbox).
- If selected person's conf_mean < conf_thr, returns cls_result=None ("no person").

API
---
from app_pose_stream import PoseStream, start_stream, read_latest, stop_stream

s = start_stream("models/.../tcn.onnx", "models/.../tcn.json", conf_thr=0.65, stride=1)
while True:
    frame, people, cls = read_latest(timeout=0.5)  # any of them can be None when not ready
    ...
stop_stream()
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU-only

import threading, queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# project modules
from . import settings as S
from .pipeline import create_pipeline_and_sink, build_pipeline
from .tcn_classifier import TCNOnnxClassifier

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

def conf_mean(person: Dict[str, Any]) -> float:
    k = person.get("kpt", [])
    vals = [float(pt[2]) for pt in k if len(pt) >= 3]
    return float(np.mean(vals)) if vals else 0.0

def bbox_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def select_person(people: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick highest mean confidence; tie-breaker = larger bbox area."""
    if not people: return None
    scored = [ (conf_mean(p), bbox_area(p["bbox"]), i, p) for i,p in enumerate(people) ]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][3]

def iou(a, b) -> float:
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
class PoseStream:
    """Start → read → stop. Returns raw frame, people, and classification result (or None)."""
    def __init__(
        self,
        onnx_path: str,
        json_path: str,
        conf_thr: float = 0.65,
        stride: int = 1,
        iou_reset_th: float = 0.30,
    ):
        self.onnx_path = onnx_path
        self.json_path = json_path
        self.conf_thr = conf_thr if conf_thr <= 1.5 else conf_thr/100.0
        self.stride = max(1, int(stride))
        self.iou_reset_th = float(iou_reset_th)

        # runtime
        self._stop = threading.Event()
        self._loop: Optional[GLib.MainLoop] = None
        self._pipe = None
        self._appsink = None
        self._data_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2)
        self._worker: Optional[threading.Thread] = None
        self._frame_i = 0

        # classifier
        self._clf: Optional[TCNOnnxClassifier] = None
        self._prev_bbox = None

        # latest result
        self._out_q: "queue.Queue[Tuple[Optional[np.ndarray], List[Dict[str, Any]], Optional[Dict[str, Any]]]]" = queue.Queue(maxsize=1)

    # Public API ---------------------------------------------------------------
    def start(self):
        Gst.init(None)
        print(build_pipeline(S)); print("==============")
        self._pipe, sink = create_pipeline_and_sink(S)
        self._appsink = sink
        try:
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
        except Exception:
            pass
        sink.connect("new-sample", self._on_new_sample, None)

        # classifier (CPU only)
        self._clf = TCNOnnxClassifier(onnx_path=self.onnx_path, json_path=self.json_path, prefer_cpu=True)
        if not (self._clf and self._clf.ok):
            print("[TCN] classifier unavailable:", getattr(self._clf, "err", None))
            self._clf = None
        else:
            print("[TCN] ready:", "NCT" if self._clf.channels_first else "NTC",
                  "| C=", self._clf.expected_C, "| T=", self._clf.expected_T,
                  "| features=", self._clf.features, "| norm=", self._clf.norm,
                  "| classes=", self._clf.classes)

        # loop + bus
        self._loop = GLib.MainLoop()
        self._attach_bus_watch(self._loop, self._pipe, "infer")

        # start pipeline
        r1 = self._pipe.set_state(Gst.State.PLAYING)
        if r1 == Gst.StateChangeReturn.FAILURE:
            print("[infer] pipeline start failed"); return

        # start worker
        self._worker = threading.Thread(target=self._app_worker, daemon=True)
        self._worker.start()

        # run GLib loop
        self._loop_thread = threading.Thread(target=self._loop.run, daemon=True)
        self._loop_thread.start()

    def read(self, timeout: Optional[float] = 0.0) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Get latest (frame, people, cls_result) tuple. Non-blocking by default.
        - frame: numpy BGR or None
        - people: list of dicts (may be empty)
        - cls_result: dict or None
            {
              "label": str, "score": float, "probs": np.ndarray,
              "conf_mean": float, "bbox": [x1,y1,x2,y2]
            }
        """
        try:
            item = self._out_q.get(timeout=timeout) if (timeout and timeout>0) else self._out_q.get_nowait()
            return item
        except queue.Empty:
            return None, [], None

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

    # Internal ----------------------------------------------------------------
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

        ok, mi = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            frame = np.frombuffer(mi.data, np.uint8).reshape((h, w, 3)).copy()
        finally:
            buf.unmap(mi)

        people = extract_people_from_buf(buf, w, h)
        pkt = {"size": (w,h), "people": people, "frame": frame}
        try:
            self._data_q.put_nowait(pkt)
        except queue.Full:
            try: _ = self._data_q.get_nowait()
            except queue.Empty: pass
            try:
                self._data_q.put_nowait(pkt)
            except queue.Full:
                pass

        return Gst.FlowReturn.OK

    def _attach_bus_watch(self, loop, pipe, name):
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
                    self._stop.set()
                    try: loop.quit()
                    except Exception: pass
            return True
        bus.add_signal_watch()
        bus.connect("message", on_msg)

    def _push_latest(self, frame, people, cls_result):
        # keep only the newest
        try:
            while True:
                self._out_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._out_q.put_nowait((frame, people, cls_result))
        except queue.Full:
            pass

    def _app_worker(self):
        while not self._stop.is_set():
            try:
                pkt = self._data_q.get(timeout=0.5)
            except queue.Empty:
                continue

            frame = pkt["frame"]
            w, h = pkt["size"]
            people = pkt["people"]

            sel = select_person(people) if people else None
            cls_result = None

            # classification (stride + conf gate)
            if self._clf is not None and sel is not None:
                cmean = conf_mean(sel)
                cur_bbox = sel.get("bbox")
                if cmean >= self.conf_thr:
                    # identity switch → reset
                    if self._prev_bbox is not None and cur_bbox is not None:
                        if iou(self._prev_bbox, cur_bbox) < self.iou_reset_th:
                            self._clf.reset()
                    self._prev_bbox = cur_bbox

                    if (self._frame_i % self.stride) == 0:
                        pred = self._clf.update([sel], (w,h))  # pass only selected
                        if pred is not None:
                            cls_result = dict(pred)
                            cls_result["conf_mean"] = float(cmean)
                            cls_result["bbox"] = list(cur_bbox) if cur_bbox is not None else None
                else:
                    # low confidence → treat as no person (also reset buffer)
                    self._clf.reset()
                    self._prev_bbox = None

            self._frame_i += 1
            self._push_latest(frame, people, cls_result)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton helpers for very simple usage
_stream_singleton: Optional[PoseStream] = None

def start_stream(onnx_path: str, json_path: str, **kwargs) -> PoseStream:
    global _stream_singleton
    if _stream_singleton is not None:
        return _stream_singleton
    _stream_singleton = PoseStream(onnx_path, json_path, **kwargs)
    _stream_singleton.start()
    return _stream_singleton

def read_latest(timeout: Optional[float] = 0.0):
    return _stream_singleton.read(timeout) if _stream_singleton else (None, [], None)

def stop_stream():
    global _stream_singleton
    if _stream_singleton is not None:
        _stream_singleton.stop()
        _stream_singleton = None