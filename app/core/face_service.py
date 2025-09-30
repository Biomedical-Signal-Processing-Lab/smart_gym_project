# app/core/face_service.py
import os, json
import numpy as np
from typing import Dict, Tuple, List, Optional
from insightface.app import FaceAnalysis

class FaceService:
    """
    - 얼굴 임베딩 추출 (InsightFace)
    - 등록/저장/로드
    - 가장 가까운 사용자 찾기
    """
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self.index_path = os.path.join(self.store_dir, "users.json")
        self.users: Dict[str, List[str]] = {}  # name -> list of .npy filenames

        # 임베딩 캐시 (탐색 속도향상)
        self._embeds: List[np.ndarray] = []
        self._owners: List[str] = []

        # InsightFace 초기화 (CPU)
        # 모델은 자동 다운로드됨 (~/.insightface)
        self.app = FaceAnalysis(name="buffalo_l")  # det + rec (arcface)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU: ctx_id=-1

        self._load_index()
        self._rebuild_cache()

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.users = json.load(f)
        else:
            self.users = {}

    def _save_index(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)

    def _rebuild_cache(self):
        self._embeds.clear()
        self._owners.clear()
        for name, files in self.users.items():
            for fn in files:
                path = os.path.join(self.store_dir, fn)
                if os.path.exists(path):
                    emb = np.load(path)
                    # 정규화(단위벡터)
                    n = np.linalg.norm(emb) + 1e-9
                    self._embeds.append(emb / n)
                    self._owners.append(name)

    # --------- Public APIs ---------
    def detect_and_embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        가장 큰 얼굴 1개 잡아서 임베딩 반환. 없으면 None
        """
        faces = self.app.get(bgr)  # [Face(...)]
        if not faces:
            return None
        # 가장 큰 bbox 선택
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding  # 이미 정규화된 L2-normalized(ArcFace)
        return emb.astype(np.float32)

    def add_user_samples(self, name: str, embeddings: List[np.ndarray]) -> None:
        """
        한 유저에 대해 여러 임베딩을 저장
        """
        safe = name.strip()
        if not safe:
            raise ValueError("빈 이름은 등록 불가")

        files = self.users.get(safe, [])
        for i, emb in enumerate(embeddings):
            fname = f"{safe}_{len(files)+i:03d}.npy"
            np.save(os.path.join(self.store_dir, fname), emb)
            files.append(fname)
        self.users[safe] = files
        self._save_index()
        self._rebuild_cache()

    def match(self, emb: np.ndarray, threshold: float = 0.35) -> Tuple[Optional[str], float]:
        """
        등록된 사용자 중 cos 유사도 최고값 반환.
        threshold는 '거리'가 아니라 1-유사도에 가까운 개념을 쓰지 않고,
        여기서는 '유사도' 컷을 0.35~0.6 범위로 사용.
        (ArcFace normed embedding에서 0.35~0.4 이상이면 같은 사람일 가능성이 꽤 높음)
        """
        if not self._embeds:
            return None, 0.0
        # emb도 정규화
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        M = np.vstack(self._embeds)  # N x D
        sims = (M @ emb)             # cosine similarity
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        owner = self._owners[idx]
        if best >= threshold:
            return owner, best
        return None, best
