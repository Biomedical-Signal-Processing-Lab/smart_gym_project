# pc_pose_stream.py
# -*- coding: utf-8 -*-
"""
PC-only (NO PyTorch): ONNXRuntime YOLOv8-Pose → unletterbox → pick largest person (kpt conf gate)
→ TCN(ONNX) classify → stream API

Public API:
    start_stream(tcn_onnx, tcn_json,
                 conf_thr=0.65, stride=1,
                 yolo_onnx="models/yolov8m_pose.onnx",
                 cam_id=None, img_size=640, det_conf=0.25, det_iou=0.50,
                 use_gpu=True, gpu_id=0, min_area=2000)
    read_latest(timeout=0.2) -> (frame, people, label)
    stop_stream()

- people: 길이 0 또는 1. 각 항목은 {"bbox":[x1,y1,x2,y2], "kpt":[(x,y,conf), ... 17]} (원본 프레임 좌표)
- label: 문자열 또는 None  (예: "(warming)", "squat", None=not detected)
"""

from __future__ import annotations
import os
import threading, queue, time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import onnxruntime as ort

try:
    import settings as S
except Exception:
    class _S:
        CAM_ID = 0
        FRAME_WIDTH = 1280
        FRAME_HEIGHT = 720
    S = _S()

from .tcn_classifier import TCNOnnxClassifier

NUM_KPTS = 17  # COCO-17

# ─────────────────────────────────────────────────────────────────────────────
# Letterbox (Ultralytics 방식 그대로) + 역변환
def _letterbox(im: np.ndarray, new_shape=640, color=(114,114,114),
               auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def _unletterbox_xyxy(xyxy: np.ndarray, ratio: float, pad: Tuple[int,int]) -> np.ndarray:
    x1,y1,x2,y2 = xyxy.astype(np.float32)
    x1 = (x1 - pad[0]) / max(ratio, 1e-6)
    y1 = (y1 - pad[1]) / max(ratio, 1e-6)
    x2 = (x2 - pad[0]) / max(ratio, 1e-6)
    y2 = (y2 - pad[1]) / max(ratio, 1e-6)
    return np.array([x1,y1,x2,y2], dtype=np.float32)

def _unletterbox_kpts(kpts: np.ndarray, ratio: float, pad: Tuple[int,int]) -> np.ndarray:
    out = kpts.copy().astype(np.float32)
    out[:,0] = (out[:,0] - pad[0]) / max(ratio, 1e-6)
    out[:,1] = (out[:,1] - pad[1]) / max(ratio, 1e-6)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8-Pose ONNX wrapper
class OnnxYoloPose:
    """
    가정: ONNX 출력이 (N,56) 혹은 (1,N,56)/(1,56,N) 디코딩 결과.
    """
    def __init__(self, onnx_path: str, img_size: int = 640, det_conf: float = 0.25, det_iou: float = 0.50,
                 use_gpu: bool = True, gpu_id: int = 0):
        self.onnx_path = onnx_path
        self.img_size  = int(img_size)
        self.det_conf  = float(det_conf)
        self.det_iou   = float(det_iou)
        avail = ort.get_available_providers()
        providers = []
        if use_gpu and "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", {"device_id": int(gpu_id)}))
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(self.onnx_path, sess_options=so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        print("[POSE] Using providers:", self.sess.get_providers())

    @staticmethod
    def _coerce_to_n56(out: np.ndarray) -> np.ndarray:
        a = out
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 2:
            if a.shape[1] == 56:
                return a
            if a.shape[0] == 56:
                return a.T
        raise RuntimeError(f"Unexpected YOLO pose ONNX output shape: {out.shape}. Expect (N,56) or (1,N,56)/(1,56,N).")

    def _pre(self, frame: np.ndarray):
        img, r, pad = _letterbox(frame, self.img_size, stride=32, auto=False, scaleFill=False, scaleup=True)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32) / 255.0
        return x[None].copy(), r, pad

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        x, r, pad = self._pre(frame)
        outs = self.sess.run(None, {self.inp: x})
        out = max(outs, key=lambda o: np.prod(o.shape))
        det = self._coerce_to_n56(out)  # (N,56)
        if det.size == 0:
            return []

        det = det[det[:,4] >= self.det_conf]
        people: List[Dict[str,Any]] = []
        for row in det:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
            xyxy_o = _unletterbox_xyxy(np.array([x1,y1,x2,y2], dtype=np.float32), r, pad)

            k = row[5:]
            if k.size == 51:
                k = k.reshape(NUM_KPTS, 3)
                k_o = _unletterbox_kpts(k, r, pad)
                pts = [(float(k_o[i,0]), float(k_o[i,1]), float(k_o[i,2])) for i in range(NUM_KPTS)]
            elif k.size == 34:
                k = k.reshape(NUM_KPTS, 2)
                k3 = np.concatenate([k, np.ones((NUM_KPTS,1),dtype=np.float32)], axis=1)
                k_o = _unletterbox_kpts(k3, r, pad)
                pts = [(float(k_o[i,0]), float(k_o[i,1]), 1.0) for i in range(NUM_KPTS)]
            else:
                continue

            people.append({
                "bbox": [int(xyxy_o[0]), int(xyxy_o[1]), int(xyxy_o[2]), int(xyxy_o[3])],
                "kpt": pts
            })
        return people

# ─────────────────────────────────────────────────────────────────────────────
def _conf_mean(person: Dict[str, Any]) -> float:
    vals = [float(p[2]) for p in person.get("kpt", []) if len(p) >= 3]
    if not vals:
        return 1.0
    m = float(np.mean(vals))
    if max(vals) <= 1e-6:
        return 1.0
    return m

def _bbox_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def _select_largest_valid(people: List[Dict[str, Any]], conf_thr: float, min_area: float):
    cands = []
    for p in people:
        cm = _conf_mean(p)
        if cm >= conf_thr and _bbox_area(p["bbox"]) >= float(min_area):
            cands.append((_bbox_area(p["bbox"]), p))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]

# ─────────────────────────────────────────────────────────────────────────────
class PCPoseStream:
    def __init__(
        self,
        tcn_onnx: str, tcn_json: str,
        conf_thr: float = 0.65, stride: int = 1,
        yolo_onnx: str = "models/yolov8m_pose.onnx",
        cam_id: Optional[int] = None,
        img_size: int = 640, det_conf: float = 0.25, det_iou: float = 0.50,
        use_gpu: bool = True, gpu_id: int = 0, min_area: float = 2000.0,
    ):
        self.tcn_onnx = tcn_onnx
        self.tcn_json = tcn_json
        self.conf_thr = conf_thr if conf_thr <= 1.5 else conf_thr/100.0
        self.stride   = max(1, int(stride))

        self.cam_id = cam_id if cam_id is not None else getattr(S, "CAM_ID", 0)
        self.cap: Optional[cv2.VideoCapture] = None

        self.det = OnnxYoloPose(yolo_onnx, img_size=img_size, det_conf=det_conf, det_iou=det_iou,
                                use_gpu=use_gpu, gpu_id=gpu_id)
        self.min_area = float(min_area)

        self.clf: Optional[TCNOnnxClassifier] = None
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._out_q: "queue.Queue[Tuple[Optional[np.ndarray], List[Dict[str, Any]], Optional[str]]]" = queue.Queue(maxsize=1)
        self._frame_i = 0

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.cam_id}")
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, getattr(S, "FRAME_WIDTH", 1280))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, getattr(S, "FRAME_HEIGHT", 720))
        except Exception:
            pass

        self.clf = TCNOnnxClassifier(self.tcn_onnx, json_path=self.tcn_json, prefer_cpu=True)
        if not (self.clf and self.clf.ok):
            raise RuntimeError(f"[TCN] init failed: {getattr(self.clf, 'err', None)}")
        print("[TCN] ready:", "NCT" if self.clf.channels_first else "NTC",
              "| C=", self.clf.expected_C, "| T=", self.clf.expected_T,
              "| features=", self.clf.features, "| norm=", self.clf.norm,
              "| classes=", self.clf.classes)

        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def read(self, timeout: Optional[float] = 0.0):
        try:
            item = self._out_q.get(timeout=timeout) if (timeout and timeout>0) else self._out_q.get_nowait()
            return item
        except queue.Empty:
            return None, [], None

    def stop(self):
        self._stop.set()
        time.sleep(0.05)
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def _publish(self, frame, people, label):
        try:
            while True:
                self._out_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._out_q.put_nowait((frame, people, label))
        except queue.Full:
            pass

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # 1) YOLO pose (원본 좌표)
            try:
                people_all = self.det.infer(frame)
            except Exception:
                people_all = []

            # 2) conf_gate + largest area
            sel = _select_largest_valid(people_all, self.conf_thr, self.min_area)
            people = [sel] if sel is not None else []

            # 3) TCN classify (+ warming 상태)
            label = None
            if self.clf is not None:
                if sel is None:
                    self.clf.reset()
                else:
                    # warming 상태 계산 (업데이트 전후 모두 체크)
                    def _is_warming():
                        try:
                            buf_len = len(self.clf.buf)
                            T_need = int(self.clf.expected_T or self.clf.win)
                            return 0 < buf_len < T_need
                        except Exception:
                            return False

                    warming = _is_warming()

                    if (self._frame_i % self.stride) == 0:
                        h, w = frame.shape[:2]
                        pred = self.clf.update([sel], (w, h))
                        if pred is not None:
                            label = str(pred.get("label", None))
                        else:
                            # 업데이트 후에도 버퍼 미충분이면 워밍
                            warming = _is_warming() or warming

                    if label is None and warming:
                        label = "(warming)"

            self._frame_i += 1
            self._publish(frame, people, label)

# ─────────────────────────────────────────────────────────────────────────────
# Singleton-style API
_stream_singleton: Optional[PCPoseStream] = None

def start_stream(
    tcn_onnx: str, tcn_json: str,
    conf_thr: float = 0.65, stride: int = 1,
    yolo_onnx: str = "models/yolov8m_pose.onnx",
    cam_id: Optional[int] = None,
    img_size: int = 640, det_conf: float = 0.25, det_iou: float = 0.50,
    use_gpu: bool = True, gpu_id: int = 0, min_area: float = 2000.0,
    **kwargs
) -> PCPoseStream:
    global _stream_singleton
    if _stream_singleton is not None:
        return _stream_singleton
    _stream_singleton = PCPoseStream(
        tcn_onnx, tcn_json,
        conf_thr=conf_thr, stride=stride,
        yolo_onnx=yolo_onnx, cam_id=cam_id,
        img_size=img_size, det_conf=det_conf, det_iou=det_iou,
        use_gpu=use_gpu, gpu_id=gpu_id, min_area=min_area,
    )
    _stream_singleton.start()
    return _stream_singleton

def read_latest(timeout: Optional[float] = 0.0):
    return _stream_singleton.read(timeout) if _stream_singleton else (None, [], None)

def stop_stream():
    global _stream_singleton
    if _stream_singleton is not None:
        _stream_singleton.stop()
        _stream_singleton = None
