# core/face_service.py
import numpy as np
from typing import List, Optional, Tuple
from db.models import User, FaceEmbedding
from insightface.app import FaceAnalysis

class FaceService:
    """
    - InsightFace로 얼굴 임베딩 추출 (CPU)
    - SQLite(DB, SQLAlchemy)에 사용자/임베딩 저장 및 조회
    - 빠른 매칭을 위한 캐시 보유
    """
    def __init__(self, SessionLocal):
        self.SessionLocal = SessionLocal

        # InsightFace: CPU 모드 (ctx_id=-1)
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

        # 캐시: (user_id, name, normalized_embed)
        self._cache: list[tuple[int, str, np.ndarray]] = []
        self._rebuild_cache()

    # --------- 내부 유틸 ---------
    def _rebuild_cache(self):
        """DB에서 모든 임베딩을 읽어 캐시(정규화) 구성"""
        self._cache.clear()
        with self.SessionLocal() as s:
            rows = (
                s.query(FaceEmbedding, User)
                .join(User, FaceEmbedding.user_id == User.id)
                .all()
            )
            for fe, user in rows:
                emb = np.frombuffer(fe.embedding, dtype=np.float32)
                n = np.linalg.norm(emb) + 1e-9
                self._cache.append((user.id, user.name, emb / n))

    # --------- 공개 API ---------
    def detect_and_embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        가장 큰 얼굴 1개를 잡아 L2-정규화 임베딩 반환. 실패 시 None.
        """
        faces = self.app.get(bgr)  # [Face(...)]
        if not faces:
            return None
        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        # ArcFace는 보통 이미 정규화되어 있음(normed_embedding)
        emb = faces[0].normed_embedding.astype(np.float32)
        return emb

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
                fe = FaceEmbedding(
                    user_id=user.id,
                    dim=int(emb.size),
                    embedding=emb.tobytes(),
                )
                s.add(fe)

            s.commit()
            uid = user.id

        # 저장 이후 캐시 갱신
        self._rebuild_cache()
        return uid


    def match(self, emb: np.ndarray, threshold: float = 0.40) -> Tuple[Optional[str], float]:
        """
        캐시에 있는 모든 임베딩과 cosine similarity 비교.
        threshold 이상이면 (name, sim) 반환, 아니면 (None, best_sim) 반환.
        """
        if not self._cache:
            return None, 0.0

        q = np.asarray(emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        best_name, best_sim = None, -1.0
        for _, name, ref in self._cache:
            sim = float(np.dot(q, ref))
            if sim > best_sim:
                best_name, best_sim = name, sim

        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim
