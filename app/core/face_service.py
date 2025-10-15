# core/face_service.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from db.models import User, FaceEmbedding
from .face_backends import FaceBackendBase, InsightFaceBackend  # , HailoFaceBackend (추후)

class FaceService:
    """
    얼굴 임베딩 추출 + DB 저장/조회 + 캐시 매칭
    - 기본: InsightFace CPU (설치 안되어 있으면 자동 비활성)
    - Hailo 백엔드(레티나+아크페이스)는 추후 같은 인터페이스로 플러그인
    """
    def __init__(self, SessionLocal, backend: FaceBackendBase | None = None):
        self.SessionLocal = SessionLocal

        # 백엔드 자동 선택: 외부에서 주입 없으면 InsightFace 시도 → 실패 시 None
        self.backend: Optional[FaceBackendBase] = backend
        if self.backend is None:
            try:
                self.backend = InsightFaceBackend(app_name="buffalo_l", det_size=(640, 640))
                self.backend.warmup()
                self._enabled = True
            except Exception as e:
                # 설치/런타임 문제가 있으면 비활성화 모드로
                print(f"[FaceService] InsightFace 사용 불가: {e}")
                self.backend = None
                self._enabled = False
        else:
            try:
                self.backend.warmup()
                self._enabled = True
            except Exception as e:
                print(f"[FaceService] 제공된 백엔드 초기화 실패: {e}")
                self.backend = None
                self._enabled = False

        self._cache: list[tuple[int, str, np.ndarray]] = []
        self._rebuild_cache()

    # 상태
    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self.backend is not None)

    # --- 캐시 관리 ---
    def _rebuild_cache(self) -> None:
        """DB 임베딩을 L2 정규화해 캐시 구성"""
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

    # --- 추론 ---
    def detect_and_embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        최대 얼굴 1개 임베딩 반환. 백엔드가 비활성화면 None.
        """
        if not self.enabled:
            return None
        try:
            return self.backend.detect_and_embed(bgr)
        except Exception as e:
            # 얼굴 기능이 앱 전체를 멈추지 않도록 방어
            print(f"[FaceService] detect_and_embed 실패: {e}")
            return None

    # --- 등록 ---
    def add_user_samples(self, name: str, embeddings: List[np.ndarray]) -> int:
        safe = (name or "").strip()
        if not safe:
            raise ValueError("빈 이름은 등록할 수 없습니다.")
        if not embeddings:
            raise ValueError("임베딩이 비어 있습니다.")

        with self.SessionLocal() as s:
            user = User(name=safe)
            s.add(user)
            s.flush()  # id 확보

            for emb in embeddings:
                emb = np.asarray(emb, dtype=np.float32)
                s.add(FaceEmbedding(user_id=user.id, dim=int(emb.size), embedding=emb.tobytes()))

            s.commit()
            uid = int(user.id)

        self._rebuild_cache()
        return uid

    # --- 매칭 ---
    def match(self, emb: np.ndarray, threshold: float = 0.40) -> tuple[Optional[str], float]:
        """
        코사인 유사도 최댓값이 threshold 이상이면 (name, sim), 아니면 (None, best_sim)
        """
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

    # --- 정리 ---
    def close(self) -> None:
        try:
            if self.backend:
                self.backend.close()
        except Exception:
            pass
