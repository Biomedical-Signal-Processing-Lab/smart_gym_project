# core/face_service.py
import numpy as np
from db.models import User, FaceEmbedding
from insightface.app import FaceAnalysis

class FaceService:
    """InsightFace로 임베딩 추출(CPU) + DB 저장/조회 + 캐시 매칭"""
    def __init__(self, SessionLocal):
        self.SessionLocal = SessionLocal
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._cache: list[tuple[int, str, np.ndarray]] = []
        self._rebuild_cache()

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
                n = np.linalg.norm(emb) + 1e-9
                self._cache.append((user.id, user.name, emb / n))

    def detect_and_embed(self, bgr: np.ndarray) -> np.ndarray | None:
        """가장 큰 얼굴 1개 임베딩(L2 정규화된 ArcFace 임베딩) 반환"""
        faces = self.app.get(bgr)
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return face.normed_embedding.astype(np.float32)

    def add_user_samples(self, name: str, embeddings: list[np.ndarray]) -> int:
        safe = (name or "").strip()
        if not safe:
            raise ValueError("빈 이름은 등록할 수 없습니다.")
        if not embeddings:
            raise ValueError("임베딩이 비어 있습니다.")

        with self.SessionLocal() as s:
            user = User(name=safe)
            s.add(user)
            s.flush()  # user.id 확보

            for emb in embeddings:
                emb = np.asarray(emb, dtype=np.float32)
                s.add(FaceEmbedding(user_id=user.id, dim=int(emb.size), embedding=emb.tobytes()))

            s.commit()
            uid = user.id

        self._rebuild_cache()
        return uid

    def match(self, emb: np.ndarray, threshold: float = 0.40) -> tuple[str | None, float]:
        """코사인 유사도 최댓값이 threshold 이상이면 (name, sim), 아니면 (None, best_sim)"""
        if not self._cache:
            return None, 0.0

        q = np.asarray(emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        best_name, best_sim = None, -1.0
        for _, name, ref in self._cache:
            sim = float(np.dot(q, ref))
            if sim > best_sim:
                best_name, best_sim = name, sim

        return (best_name, best_sim) if best_sim >= threshold else (None, best_sim)
