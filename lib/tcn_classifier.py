# tcn_classifier.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

NUM_KPTS = 17

def _softmax_axis(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

class TCNOnnxClassifier:
    """
    JSON 예시:
    {
      "classes": ["idle","plank","pushup","shoulder_press","squat"],
      "input_channels": 51,          # xy=34, xyconf=51 (학습과 동일)
      "features": "xyconf",          # "xy" or "xyconf"
      "norm": "image",               # "image" or "bbox"
      "hparams": {"win": 60},
      "kpt_order": null,             # 선택: 길이 17 인덱스
      "smooth": 7                    # EMA 윈도우(>1이면 적용)
    }
    """
    def __init__(self, onnx_path: str, json_path: str = "/mnt/data/tcn.json", prefer_cpu: bool = True):
        self.ok: bool = False
        self.err: Optional[Exception] = None

        # 1) JSON 로드
        cfg: Dict[str, Any] = {}
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        self.classes: Optional[List[str]] = cfg.get("classes")
        self.features: str = cfg.get("features", "xyconf")
        self.norm: str = cfg.get("norm", "image")
        hp = cfg.get("hparams", {}) or {}
        self.win: int = int(hp.get("win", 60))
        self.input_channels_cfg: int = int(cfg.get("input_channels", 51 if self.features == "xyconf" else 34))
        self.kpt_order: Optional[List[int]] = cfg.get("kpt_order")
        if self.kpt_order is not None:
            if not isinstance(self.kpt_order, list) or len(self.kpt_order) != NUM_KPTS:
                self.kpt_order = None

        # EMA smoothing
        self.smooth: int = int(cfg.get("smooth", 7))
        self._ema: Optional[np.ndarray] = None
        self._alpha: float = 2.0/float(self.smooth+1) if self.smooth and self.smooth > 1 else 1.0

        # 2) 버퍼
        self.buf: deque = deque(maxlen=self.win)

        # 3) ONNX 세션
        self.session = None
        self.input_name: Optional[str] = None
        self.channels_first: Optional[bool] = None  # True:[N,C,T], False:[N,T,C]
        self.expected_C: Optional[int] = None
        self.expected_T: Optional[int] = None
        try:
            if ort is None:
                raise RuntimeError("onnxruntime is not installed")
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            providers = ["CPUExecutionProvider"] if prefer_cpu else None
            self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self._infer_layout_and_dims()
            # PT와 동일하게 N,C,T 강제
            self.channels_first = True
            # 모델 T가 더 크면 버퍼 크기 보정
            if isinstance(self.expected_T, int) and self.expected_T > self.win:
                self.win = self.expected_T
                self.buf = deque(maxlen=self.win)
            self.ok = True
        except Exception as e:
            self.err = e
            self.ok = False

    # ──────────────────────────────────────────────────────────────
    def _infer_layout_and_dims(self):
        C_json = int(self.input_channels_cfg)
        T_json = int(self.win)

        def try_layout(ch_first: bool):
            T_try = T_json
            x = np.zeros((1, C_json, T_try), np.float32) if ch_first else np.zeros((1, T_try, C_json), np.float32)
            try:
                _ = self.session.run(None, {self.input_name: x})
                return True, T_try
            except Exception:
                return False, None

        ok_nct, T_nct = try_layout(True)
        ok_ntc, T_ntc = try_layout(False)

        # 감지 결과와 무관하게 C/T는 기록
        self.expected_T = T_nct if ok_nct else (T_ntc if ok_ntc else T_json)
        self.expected_C = C_json

        # 출력 클래스 수와 classes 정합
        out0 = self.session.get_outputs()[0]
        oshape = out0.shape  # [N, C] 또는 [N, C, T] 등
        if len(oshape) >= 2 and isinstance(oshape[1], int) and oshape[1] > 0:
            n_cls = oshape[1]
            if not self.classes or len(self.classes) != n_cls:
                self.classes = (self.classes or [])[:n_cls]
                while len(self.classes) < n_cls:
                    self.classes.append(f"class_{len(self.classes)}")

    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _select_person(people: List[Dict]) -> Optional[Dict]:
        """키포인트 confidence 평균이 가장 높은 사람(동률 시 면적 큰 사람)."""
        if not people: return None
        def conf_mean(p):
            k = p.get("kpt", [])
            if not k: return -1.0
            vals = [float(pt[2]) for pt in k if len(pt) >= 3]
            return float(np.mean(vals)) if vals else -1.0
        def area(p):
            x1,y1,x2,y2 = p["bbox"]
            return max(0, x2-x1) * max(0, y2-y1)
        return max(people, key=lambda p: (conf_mean(p), area(p)))

    def _reorder_kpts(self, pts: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        if self.kpt_order is None: return pts
        out = [(0,0)] * NUM_KPTS
        for i_src, i_dst in enumerate(self.kpt_order):
            if i_src < len(pts) and 0 <= i_dst < NUM_KPTS:
                out[i_dst] = pts[i_src]
        return out

    def _norm_xy(self, x: float, y: float, w: int, h: int, bbox: Optional[List[int]]) -> Tuple[float, float]:
        if self.norm == "bbox" and bbox is not None:
            x1,y1,x2,y2 = bbox
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            return ((x - x1) / bw, (y - y1) / bh)
        return (x / max(1.0, float(w)), y / max(1.0, float(h)))  # image

    def _feat_xy(self, person: Dict, size: Tuple[int,int]) -> np.ndarray:
        w, h = size
        pts = self._reorder_kpts(person.get("kpt", []))
        out = np.zeros((NUM_KPTS, 2), dtype=np.float32)
        bbox = person.get("bbox", None)
        for i in range(min(NUM_KPTS, len(pts))):
            x, y = pts[i] if len(pts[i]) >= 2 else (0, 0)
            xn, yn = self._norm_xy(float(x), float(y), w, h, bbox)
            out[i] = (xn, yn)
        return out.reshape(-1)  # (34,)

    def _feat_xyconf(self, person: Dict, size: Tuple[int,int]) -> np.ndarray:
        """PT와 동일한 패킹: [x0,y0,x1,y1,...,y16, c0..c16]"""
        w, h = size
        pts = self._reorder_kpts(person.get("kpt", []))
        bbox = person.get("bbox", None)

        X = np.zeros(17, np.float32)
        Y = np.zeros(17, np.float32)
        C = np.ones(17, np.float32)

        for i, p in enumerate(pts[:17]):
            if len(p) >= 3: x, y, c = p[0], p[1], float(p[2])
            elif len(p) >= 2: x, y, c = p[0], p[1], 1.0
            else: x, y, c = 0.0, 0.0, 1.0
            xn, yn = self._norm_xy(float(x), float(y), w, h, bbox)
            X[i], Y[i], C[i] = xn, yn, c

        xy_interleaved = np.empty(34, np.float32)
        xy_interleaved[0::2] = X
        xy_interleaved[1::2] = Y
        return np.concatenate([xy_interleaved, C], axis=0)  # (51,)

    def _make_feat_vec(self, person: Dict, size: Tuple[int,int]) -> np.ndarray:
        if self.features == "xy":
            vec = self._feat_xy(person, size)
        else:
            vec = self._feat_xyconf(person, size)

        C = int(self.expected_C or self.input_channels_cfg or len(vec))
        if len(vec) != C:
            tmp = np.zeros((C,), dtype=np.float32)
            n = min(C, len(vec))
            tmp[:n] = vec[:n]
            vec = tmp
        return vec

    # ──────────────────────────────────────────────────────────────
    def reset(self): 
        self.buf.clear()
        self._ema = None

    def update(self, people: List[Dict], size: Tuple[int,int]) -> Optional[Dict]:
        if not self.ok or self.session is None: return None

        person = self._select_person(people)  # conf-우선 선택
        if person is None:
            self.reset()
            return None

        feat = self._make_feat_vec(person, size)  # (C,)
        self.buf.append(feat)

        T_need = int(self.expected_T or self.win)
        if len(self.buf) < T_need:
            return None

        window = np.stack(list(self.buf)[-T_need:], axis=0)  # (T, C)
        # PT와 동일: N,C,T
        x = np.transpose(window[None, ...], (0, 2, 1)).astype(np.float32)

        try:
            out = self.session.run(None, {self.input_name: x})
        except Exception:
            return None

        logits = out[0].astype(np.float32)
        # 출력축 안전 처리
        if logits.ndim == 2:  # (N, C)
            probs = _softmax_axis(logits, axis=1)
        elif logits.ndim == 3:
            # (N, C, T) 또는 (N, T, C)
            if self.classes and logits.shape[1] == len(self.classes):
                probs_t = _softmax_axis(logits, axis=1)  # 클래스축
                probs = probs_t.mean(axis=2)             # 시간 평균 → (N, C)
            elif self.classes and logits.shape[2] == len(self.classes):
                probs_t = _softmax_axis(logits, axis=2)
                probs = probs_t.mean(axis=1)
            else:
                probs = _softmax_axis(logits, axis=-1).mean(axis=-2)
        else:
            probs = _softmax_axis(logits, axis=-1)

        # classes 보정
        n_cls = probs.shape[-1]
        if not self.classes or len(self.classes) != n_cls:
            self.classes = (self.classes or [])[:n_cls]
            while len(self.classes) < n_cls:
                self.classes.append(f"class_{len(self.classes)}")

        # EMA 스무딩 (선택)
        probs_1d = probs[0]
        if self.smooth and self.smooth > 1:
            if self._ema is None: self._ema = probs_1d
            else: self._ema = (1 - self._alpha) * self._ema + self._alpha * probs_1d
            probs_1d = self._ema

        idx = int(np.argmax(probs_1d))
        score = float(probs_1d[idx])
        return {"label": self.classes[idx], "score": score, "probs": probs_1d}