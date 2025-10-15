# core/hailo_cam_adapter.py
from __future__ import annotations
import math, threading
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from . import settings as S
from .hailo_pose_stream import start_stream, read_latest, stop_stream

# YOLOv8 Pose index
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

def _angle(a, b, c) -> Optional[float]:
    if a is None or b is None or c is None: return None
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax-bx, ay-by], dtype=np.float32)
    v2 = np.array([cx-bx, cy-by], dtype=np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    cosv = float(np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))

class HailoCamAdapter:
    def __init__(self, conf_thr: float = 0.65, stride: int = 1):
        self.conf_thr = conf_thr if conf_thr <= 1.5 else conf_thr/100.0
        self.stride = int(max(1, stride))
        self._lock = threading.Lock()
        self._frame_rgb: Optional[np.ndarray] = None
        self._people: List[Dict[str, Any]] = []
        self._cls: Optional[Dict[str, Any]] = None
        self._size: Tuple[int,int] = (S.SRC_WIDTH, S.SRC_HEIGHT)
        self._running = False

    def start(self):
        if self._running: return
        start_stream(conf_thr=self.conf_thr, stride=self.stride)
        self._running = True

    def stop(self):
        if not self._running: return
        stop_stream()
        self._running = False

    def _pull_once(self) -> bool:
        fr, people, cls, size = read_latest(timeout=0.01)
        if fr is None: return False
        with self._lock:
            self._frame_rgb = fr
            self._people = people
            self._cls = cls
            self._size = size
        return True

    def frame(self) -> Optional[np.ndarray]:
        self._pull_once()
        with self._lock:
            return None if self._frame_rgb is None else self._frame_rgb.copy()

    def people(self) -> List[Dict[str, Any]]:
        self._pull_once()
        with self._lock:
            return list(self._people)

    def meta(self) -> Dict[str, Any]:
        """UI에서 쓰는 상태 메타"""
        self._pull_once()
        with self._lock:
            ok = bool(self._people)
            w, h = self._size
            label = self._cls.get("label") if isinstance(self._cls, dict) else None

            knees = (None, None)
            if self._people:
                p = self._people[0]  # 신뢰도 높은 1명
                pts = p.get("kpt", [])
                def pt(idx):
                    if idx >= len(pts): return None
                    x, y = int(pts[idx][0]), int(pts[idx][1])
                    c = float(pts[idx][2]) if len(pts[idx]) >= 3 else 1.0
                    return (x,y) if c >= self.conf_thr else None
                l_ang = _angle(pt(L_HIP), pt(L_KNEE), pt(L_ANK))
                r_ang = _angle(pt(R_HIP), pt(R_KNEE), pt(R_ANK))
                knees = (l_ang, r_ang)

            return {
                "ok": ok,
                "src_w": w, "src_h": h,
                "label": label,                # ex) "squat" (TCN 사용 시)
                "knee_l_deg": knees[0],
                "knee_r_deg": knees[1],
            }
