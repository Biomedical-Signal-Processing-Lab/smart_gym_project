# core/pose_manager.py
import time
from typing import Optional, Tuple, List
import cv2
import numpy as np
import onnxruntime as ort
from threading import Lock

KPT = {
    "L_SHO":5, "R_SHO":6, "L_ELB":7, "R_ELB":8, "L_WRIST":9, "R_WRIST":10,
    "L_HIP":11, "R_HIP":12, "L_KNEE":13, "R_KNEE":14, "L_ANK":15, "R_ANK":16
}

PAIRS = [
    (5,7),(7,9), (6,8),(8,10),          # 팔
    (11,13),(13,15), (12,14),(14,16),   # 다리
    (5,6), (5,11),(6,12), (11,12)       # 몸통
]

def _pick_first_valid(out_list):
    if isinstance(out_list, (list, tuple)):
        for o in out_list:
            if isinstance(o, np.ndarray) and o.ndim == 3 and o.shape[0] == 1:
                return o
        for o in out_list:
            if isinstance(o, np.ndarray):
                return o
        return out_list[0]
    return out_list

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))

def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_th: float = 0.45) -> List[int]:
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        inter = np.maximum(0.0, xx2-xx1) * np.maximum(0.0, yy2-yy1)
        area_i = (boxes[i,2]-boxes[i,0]) * (boxes[i,3]-boxes[i,1])
        area_j = (boxes[idxs[1:],2]-boxes[idxs[1:],0]) * (boxes[idxs[1:],3]-boxes[idxs[1:],1])
        ovr = inter / (area_i + area_j - inter + 1e-9)
        idxs = idxs[1:][ovr <= iou_th]
    return keep

class PoseProcessor:
    def __init__(
        self,
        onnx_path: str = "models/yolov8m_pose.onnx",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        providers: List[str] | None = None,
        draw_landmarks: bool = True,
        name: str = "pose",
    ):
        self.name = name
        self.conf_thres = float(conf_thres)
        self.iou_thres  = float(iou_thres)
        self._draw_landmarks = bool(draw_landmarks)
        self._meta = {"ok": False}
        self._lock = Lock()

        avail = set(ort.get_available_providers())
        pref = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        final = [p for p in pref if p in avail] or ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=final)

        inp0 = self.sess.get_inputs()[0]
        self.input_name = inp0.name
        h, w = inp0.shape[2], inp0.shape[3]
        self._in_h = int(h) if h is not None else 640
        self._in_w = int(w) if w is not None else 640

    def set_draw_landmarks(self, enabled: bool):
        with self._lock:
            self._draw_landmarks = bool(enabled)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame_bgr, (self._in_w, self._in_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0)[None, ...]
        return img

    def _postprocess(self, out: np.ndarray, orig_shape: Tuple[int,int]):
        if not (isinstance(out, np.ndarray) and out.ndim == 3 and out.shape[0] == 1):
            return []
        pred = out[0].T if out.shape[1] < out.shape[2] else out[0]  
        if pred.shape[1] < 56: 
            return []

        boxes_xywh = pred[:, :4]               
        scores     = pred[:, 4]
        kpts       = pred[:, 5:5+17*3].reshape(-1, 17, 3)  

        # 정규화 방어 (0~1이면 입력크기로 확장)
        if np.all(kpts[..., :2] >= -1e-6) and np.all(kpts[..., :2] <= 1.000001):
            kpts[..., 0] *= float(self._in_w)
            kpts[..., 1] *= float(self._in_h)
        if np.all(boxes_xywh >= -1e-6) and np.all(boxes_xywh <= 1.000001):
            boxes_xywh[:, [0,2]] *= float(self._in_w)
            boxes_xywh[:, [1,3]] *= float(self._in_h)

        mask = scores >= self.conf_thres
        if not np.any(mask): return []
        boxes_xywh, scores, kpts = boxes_xywh[mask], scores[mask], kpts[mask]

        cx, cy, w, h = boxes_xywh[:,0], boxes_xywh[:,1], boxes_xywh[:,2], boxes_xywh[:,3]
        xyxy = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)

        keep = _nms_boxes(xyxy, scores, self.iou_thres)
        if not keep: return []
        xyxy, scores, kpts = xyxy[keep], scores[keep], kpts[keep]

        H, W = orig_shape
        sx, sy = float(W)/self._in_w, float(H)/self._in_h
        xyxy[:, [0,2]] *= sx; xyxy[:, [1,3]] *= sy
        kpts[..., 0]   *= sx; kpts[..., 1]   *= sy

        xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, W-1)
        xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, H-1)
        kpts[..., 0]   = np.clip(kpts[..., 0], 0, W-1)
        kpts[..., 1]   = np.clip(kpts[..., 1], 0, H-1)

        return [(float(scores[i]), xyxy[i], kpts[i]) for i in range(len(keep))]

    def _draw(self, img: np.ndarray, kpts_xyc: np.ndarray):
        for i in range(17):
            x, y, c = kpts_xyc[i]
            if c >= 0.05:
                cv2.circle(img, (int(x), int(y)), 3, (0,255,255), -1, lineType=cv2.LINE_AA)
        # 라인(얼굴 제외)
        for a, b in PAIRS:
            xa, ya, ca = kpts_xyc[a]
            xb, yb, cb = kpts_xyc[b]
            if ca >= 0.05 and cb >= 0.05:
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), (0,255,0), 2, lineType=cv2.LINE_AA)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        out_img = frame_bgr.copy()
        H, W = out_img.shape[:2]

        inp = self._preprocess(frame_bgr)
        outs = self.sess.run(None, {self.input_name: inp})
        out  = _pick_first_valid(outs)
        dets = self._postprocess(out, (H, W))

        if not dets:
            with self._lock:
                self._meta = dict(ok=False, ts=time.time(),
                                  hip_y_px=None, hip_y_norm=None,
                                  knee_l_deg=None, knee_r_deg=None,
                                  elbow_l_deg=None, elbow_r_deg=None)
            return out_img

        _, _, k = max(dets, key=lambda d: d[0])

        if self._draw_landmarks:
            self._draw(out_img, k)

        def pt(i): 
            x, y, _ = k[i]; 
            return np.array([float(x), float(y)], dtype=np.float32)

        try:
            # 무릎
            ang_knee_l = _angle_deg(pt(KPT["L_HIP"]), pt(KPT["L_KNEE"]), pt(KPT["L_ANK"]))
            ang_knee_r = _angle_deg(pt(KPT["R_HIP"]), pt(KPT["R_KNEE"]), pt(KPT["R_ANK"]))
            # 팔꿈치
            ang_elb_l  = _angle_deg(pt(KPT["L_SHO"]), pt(KPT["L_ELB"]), pt(KPT["L_WRIST"]))
            ang_elb_r  = _angle_deg(pt(KPT["R_SHO"]), pt(KPT["R_ELB"]), pt(KPT["R_WRIST"]))
            # 엉덩이 높이
            hip_mid_y_px = float((k[KPT["L_HIP"],1] + k[KPT["R_HIP"],1]) * 0.5)
            hip_mid_y_norm = hip_mid_y_px / float(H)

            meta = {
                "ok": True, "ts": time.time(),
                "hip_y_px": hip_mid_y_px,
                "hip_y_norm": hip_mid_y_norm,
                "knee_l_deg": None if ang_knee_l is None else float(ang_knee_l),
                "knee_r_deg": None if ang_knee_r is None else float(ang_knee_r),
                "elbow_l_deg": None if ang_elb_l is None else float(ang_elb_l),
                "elbow_r_deg": None if ang_elb_r is None else float(ang_elb_r),
            }
        except Exception:
            meta = dict(ok=False, ts=time.time(),
                        hip_y_px=None, hip_y_norm=None,
                        knee_l_deg=None, knee_r_deg=None,
                        elbow_l_deg=None, elbow_r_deg=None)

        with self._lock:
            self._meta = meta
        return out_img

    def get_meta(self):
        with self._lock:
            return dict(self._meta)

    def close(self):
        self.sess = None
