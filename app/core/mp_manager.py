# core/mp_manager.py
import cv2
import time 
import numpy as np
import mediapipe as mp
from threading import Lock

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def _angle_deg(a, b, c):
    # a(hip) - b(knee) - c(ankle) 에서 무릎 각도 계산
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

class PoseProcessor:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        draw_landmarks: bool = True,
        draw_segmentation: bool = False,
        name: str = "pose"
    ):
        self.name = name
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.draw_landmarks = draw_landmarks
        self.draw_segmentation = draw_segmentation

        self._meta = {"ok": False}
        self._lock = Lock()

        self.idx = mp_pose.PoseLandmark
        self._LHIP  = int(self.idx.LEFT_HIP)
        self._RHIP  = int(self.idx.RIGHT_HIP)
        self._LKNEE = int(self.idx.LEFT_KNEE)
        self._RKNEE = int(self.idx.RIGHT_KNEE)
        self._LANK  = int(self.idx.LEFT_ANKLE)
        self._RANK  = int(self.idx.RIGHT_ANKLE)

    def _to_px(self, lm, w, h):
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(img_rgb)

        out = frame_bgr

        if self.draw_segmentation and getattr(res, "segmentation_mask", None) is not None:
            mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(out, 0.7, mask, 0.3, 0)

        if self.draw_landmarks and getattr(res, "pose_landmarks", None):
            mp_drawing.draw_landmarks(
                image=out,
                landmark_list=res.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
            )

        meta = {
            "ok": False,
            "ts": time.time(),
            "hip_y_px": None,
            "hip_y_norm": None,
            "knee_l_deg": None,
            "knee_r_deg": None,
        }

        lms = getattr(res, "pose_landmarks", None)
        if lms:
            lms = lms.landmark
            try:
                hipL  = lms[self._LHIP];  hipR  = lms[self._RHIP]
                kneeL = lms[self._LKNEE]; kneeR = lms[self._RKNEE]
                ankL  = lms[self._LANK];  ankR  = lms[self._RANK]

                hip_mid_y_norm = (hipL.y + hipR.y) / 2.0
                hip_mid_y_px   = hip_mid_y_norm * h

                H_L = self._to_px(hipL, w, h); K_L = self._to_px(kneeL, w, h); A_L = self._to_px(ankL, w, h)
                H_R = self._to_px(hipR, w, h); K_R = self._to_px(kneeR, w, h); A_R = self._to_px(ankR, w, h)

                angL = _angle_deg(H_L, K_L, A_L)
                angR = _angle_deg(H_R, K_R, A_R)

                meta.update({
                    "ok": True,
                    "hip_y_px":   float(hip_mid_y_px),
                    "hip_y_norm": float(hip_mid_y_norm),
                    "knee_l_deg": None if angL is None else float(angL),
                    "knee_r_deg": None if angR is None else float(angR),
                })
            except Exception:
                pass

        with self._lock:
            self._meta = meta

        return out

    def get_meta(self):
        with self._lock:
            return dict(self._meta)

    def close(self):
        if self.pose:
            self.pose.close()
            self.pose = None
