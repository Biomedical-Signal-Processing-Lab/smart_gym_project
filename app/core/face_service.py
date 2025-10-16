from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple, Optional
from db.models import User, FaceEmbedding
from . import settings as S
from .face_backends import FaceBackendBase, HailoFaceBackend

_ARC_STD_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def align_by_5pts(bgr: np.ndarray, kpt5: List[Tuple[int, int]], out_size: Tuple[int,int] = (112, 112)) -> np.ndarray:
    src = np.array(kpt5, dtype=np.float32)
    dst = _ARC_STD_5PTS.copy()
    if out_size != (112, 112):
        sx = out_size[0] / 112.0
        sy = out_size[1] / 112.0
        dst[:, 0] *= sx
        dst[:, 1] *= sy
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        x = int(min(p[0] for p in kpt5)); y = int(min(p[1] for p in kpt5))
        X = int(max(p[0] for p in kpt5)); Y = int(max(p[1] for p in kpt5))
        x = max(0, x); y = max(0, y)
        crop = bgr[y:Y, x:X].copy() if (Y > y and X > x) else bgr
        return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(bgr, M, out_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

class FaceService:
    def __init__(self, SessionLocal, backend: FaceBackendBase | None = None):
        self.SessionLocal = SessionLocal
        self.backend: Optional[FaceBackendBase] = backend
        self._enabled = False

        try:
            if self.backend is None:
                self.backend = HailoFaceBackend(
                    det_hef=getattr(S, "FACE_DET_HEF", None),
                    post_so=getattr(S, "FACE_POST_SO", None),
                    cropper_so=getattr(S, "CROPPER_SO", None),
                    cam=getattr(S, "CAM", None),
                    det_input_size=(S.SRC_WIDTH, S.SRC_HEIGHT),
                    arcface_app=S.ARC_APP_NAME,
                    arcface_input=getattr(S, "ARC_INPUT_HW", (112, 112)),
                )
            self._enabled = True
        except Exception:
            self.backend = None
            self._enabled = False

        self._cache = []
        self._rebuild_cache()

    def start_stream(self):
        if self.enabled and self.backend:
            try: self.backend.start()
            except Exception: pass

    def stop_stream(self):
        if self.backend:
            try: self.backend.stop()
            except Exception: pass

    @property
    def enabled(self) -> bool:
        return bool(getattr(self, "_enabled", False) and self.backend is not None)

    def _rebuild_cache(self) -> None:
        self._cache.clear()
        with self.SessionLocal() as s:
            rows = (
                s.query(FaceEmbedding, User)
                 .join(User, FaceEmbedding.user_id == User.id)
                 .all()
            )
            for fe, user in rows:
                emb = np.frombuffer(fe.embedding, dtype=np.float32)
                n = float(np.linalg.norm(emb)) + 1e-9
                self._cache.append((int(user.id), str(user.name), emb / n))

    def detect_and_embed(self, bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        try:
            return self.backend.detect_and_embed(bgr)
        except Exception:
            return None

    def add_user_samples(self, name: str, embeddings: List[np.ndarray]) -> int:
        safe = (name or "").strip()
        if not safe:
            raise ValueError("빈 이름은 등록할 수 없습니다.")
        if not embeddings:
            raise ValueError("임베딩이 비어 있습니다.")

        with self.SessionLocal() as s:
            user = User(name=safe)
            s.add(user)
            s.flush()

            for emb in embeddings:
                emb = np.asarray(emb, dtype=np.float32)
                s.add(FaceEmbedding(user_id=user.id, dim=int(emb.size), embedding=emb.tobytes()))

            s.commit()
            uid = int(user.id)

        self._rebuild_cache()
        return uid

    def match(self, emb: np.ndarray, threshold: float = S.FACE_MATCH_THRESHOLD) -> tuple[Optional[str], float]:
        if not self._cache:
            return None, 0.0

        q = np.asarray(emb, dtype=np.float32)
        q = q / (float(np.linalg.norm(q)) + 1e-9)

        best_name, best_sim = None, -1.0
        for _uid, name, ref in self._cache:
            sim = float(np.dot(q, ref))
            if sim > best_sim:
                best_name, best_sim = name, sim

        return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)

    def close(self) -> None:
        try:
            if self.backend:
                self.backend.close()
        except Exception:
            pass
