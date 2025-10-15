# core/face_backends.py
from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np

class FaceBackendBase:
    """얼굴 검출+임베딩 공통 인터페이스"""
    def warmup(self) -> None: ...
    def detect_and_embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """최대 얼굴 1개 임베딩(float32) 또는 None"""
        raise NotImplementedError
    def close(self) -> None: ...

# -------- InsightFace (CPU) --------
class InsightFaceBackend(FaceBackendBase):
    def __init__(self, app_name: str = "buffalo_l", det_size=(640, 640)):
        try:
            from insightface.app import FaceAnalysis  # 지연 임포트
        except Exception as e:
            raise RuntimeError("insightface가 설치되지 않았습니다.") from e
        self.FaceAnalysis = FaceAnalysis
        self.app_name = app_name
        self.det_size = det_size
        self.app = None

    def warmup(self) -> None:
        self.app = self.FaceAnalysis(name=self.app_name)
        # ctx_id=-1 -> CPU 강제
        self.app.prepare(ctx_id=-1, det_size=self.det_size)

    def detect_and_embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.app is None:
            self.warmup()
        faces = self.app.get(bgr)  # bgr 입력 OK
        if not faces:
            return None
        # 가장 큰 얼굴
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding.astype(np.float32, copy=False)
        return emb

    def close(self) -> None:
        self.app = None

# -------- Hailo (RetinaFace+ArcFace) 자리(추후) --------
class HailoFaceBackend(FaceBackendBase):
    """
    TODO: retinaface_xxx.hef + arcface_xxx.hef 준비되면 구현.
    현재는 명시적으로 미구현 에러를 던져 개발 중임을 알림.
    """
    def __init__(self, retina_hef: str, arcface_hef: str):
        self.retina_hef = retina_hef
        self.arcface_hef = arcface_hef

    def warmup(self) -> None:
        raise NotImplementedError("Hailo 얼굴 백엔드는 아직 구현되지 않았습니다.")

    def detect_and_embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError("Hailo 얼굴 백엔드는 아직 구현되지 않았습니다.")
