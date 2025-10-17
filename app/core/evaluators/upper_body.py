# app/core/evaluators/upper_body.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from PySide6.QtGui import QColor
from .base import ExerciseEvaluator, EvalResult


# -------------------- 공용 유틸 --------------------
def _get_first(meta: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = meta.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def _avg_lr(meta: Dict[str, Any], base: str) -> Optional[float]:
    variants = {
        "shoulder": [["shoulder_l","Shoulder(L)"], ["shoulder_r","Shoulder(R)"]],
        "elbow":    [["elbow_l","Elbow(L)"], ["elbow_r","Elbow(R)"]],
        "knee":     [["knee_l","Knee(L)"], ["knee_r","Knee(R)"]],
    }
    if base not in variants:
        return None
    l = _get_first(meta, variants[base][0])
    r = _get_first(meta, variants[base][1])
    if l is None and r is None: return None
    if l is None: return r
    if r is None: return l
    return (l + r) / 2.0


def _right_only(meta: Dict[str, Any], base: str) -> Optional[float]:
    """오른쪽 관절만 사용"""
    return _get_first(meta, [f"{base}_r", f"{base}_R", f"{base}(R)", base.capitalize()+"(R)"])


# -------------------- UpperBodyEvaluator --------------------
class UpperBodyEvaluator(ExerciseEvaluator):
    """
    푸쉬업 / 숄더 프레스 / 사이드 레터럴 레이즈 / 덤벨 로우
    - 각 운동별 카운트 기준과 점수 산출 시점을 분리
    - 카운트: 동작 한 사이클 인식
    - 점수 : 특정 관절 조건 충족 시 스냅샷 평가
    """
    DEBOUNCE_N = 3
    TOL = 3.0

    # 각도 기준 테이블
    THRESHOLDS = {
        "pushup": {
            "shoulder_r": {"best": 38.0, "max": 45.0, "min": 30.0},
            "elbow_r":    {"best": 130.0, "max": 180.0, "min": 130.0},
            "knee_r":     {"best": 157.0, "max": 157.0, "min": 90.0},
        },
        "shoulder_press": {
            "shoulder": {"best": 153.0, "max": 153.0, "min": 140.0},
            "elbow":    {"best": 143.0, "max": 143.0, "min": 137.0},
        },
        "side_lateral_raise": {
            "shoulder": {"best": 87.0, "max": 99.5, "min": 58.0},
            "elbow":    {"best": 176.5, "max": 180.0, "min": 130.0},
        },
        "dumbbell_row": {
            "shoulder": {"best": 30.0,  "max": 45.0, "min": 18.0},
            "elbow":    {"best": 120.0, "max": 165.0, "min": 145.0},
        },
    }

    def __init__(self, label: str):
        super().__init__()
        # 라벨 정규화(팩토리와 동일 기준)
        self.mode = {"Side_lateral_raise":"side_lateral_raise",
                     "Dumbbell_Row":"dumbbell_row"}.get(label, label)
        self.reset()

    def reset(self):
        self.state = "UP"
        self._deb = 0

    # ✅ ExercisePage가 호출하는 표준 엔트리
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        m = self.mode
        if m == "pushup":
            return self._update_pushup(meta)
        if m == "shoulder_press":
            return self._update_shoulder_press(meta)
        if m == "side_lateral_raise":
            return self._update_side_lateral(meta)
        if m == "dumbbell_row":
            return self._update_dumbbell_row(meta)
        return None

    # -------------------- 공용 점수 계산 --------------------
    def _score_by_angle(self, angle: float, cfg: Dict[str, float]) -> int:
        b, mx, mn = cfg["best"], cfg["max"], cfg["min"]
        if abs(angle - b) <= self.TOL:
            return 100
        if mx <= angle < b:
            return 80
        if b < angle <= mn:
            return 80
        return 50

    def _snapshot_score(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        """현재 프레임에서 관련 관절 평균점수 계산"""
        used: Dict[str, float] = {}
        scores: List[int] = []
        cfgs = self.THRESHOLDS[self.mode]

        if self.mode == "pushup":
            s = _right_only(meta, "shoulder")
            e = _right_only(meta, "elbow")
            k = _right_only(meta, "knee")
            if s is not None:
                used["shoulder_r"] = s; scores.append(self._score_by_angle(s, cfgs["shoulder_r"]))
            if e is not None:
                used["elbow_r"] = e; scores.append(self._score_by_angle(e, cfgs["elbow_r"]))
            if k is not None:
                used["knee_r"] = k; scores.append(self._score_by_angle(k, cfgs["knee_r"]))
            advice = "팔과 몸의 각도를 일정하게 유지하세요."

        elif self.mode == "shoulder_press":
            s = _avg_lr(meta, "shoulder"); e = _avg_lr(meta, "elbow")
            if s is not None:
                used["shoulder"] = s; scores.append(self._score_by_angle(s, cfgs["shoulder"]))
            if e is not None:
                used["elbow"] = e; scores.append(self._score_by_angle(e, cfgs["elbow"]))
            advice = "팔을 천천히 위로 올리세요."

        elif self.mode == "side_lateral_raise":
            s = _avg_lr(meta, "shoulder"); e = _avg_lr(meta, "elbow")
            if s is not None:
                used["shoulder"] = s; scores.append(self._score_by_angle(s, cfgs["shoulder"]))
            if e is not None:
                used["elbow"] = e; scores.append(self._score_by_angle(e, cfgs["elbow"]))
            advice = "팔이 너무 높지 않게 들어주세요."

        elif self.mode == "dumbbell_row":
            s = _avg_lr(meta, "shoulder"); e = _avg_lr(meta, "elbow")
            if s is not None:
                used["shoulder"] = s; scores.append(self._score_by_angle(s, cfgs["shoulder"]))
            if e is not None:
                used["elbow"] = e; scores.append(self._score_by_angle(e, cfgs["elbow"]))
            advice = "등 근육으로 팔을 당기세요."

        score = int(round(sum(scores) / max(1, len(scores))))
        return score, used, advice

    # -------------------- 메인 업데이트 --------------------
    def update_and_maybe_score(self, meta: Dict[str, Any], label: Optional[str] = None) -> Optional[EvalResult]:
        if label:
            self.mode = label

        if self.mode == "pushup":
            return self._update_pushup(meta)
        if self.mode == "shoulder_press":
            return self._update_shoulder_press(meta)
        if self.mode == "side_lateral_raise":
            return self._update_side_lateral(meta)
        if self.mode == "dumbbell_row":
            return self._update_dumbbell_row(meta)
        return None

    # -------------------- 1. PUSH-UP --------------------
    # 숫자 새는 기준: 어깨가 내려갔다가 올라올 때
    # 점수 시점: 오른쪽 엘보우가 130 이상일 때
    def _update_pushup(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        shoulder = _right_only(meta, "shoulder")
        elbow = _right_only(meta, "elbow")
        if shoulder is None or elbow is None:
            return None

        # 카운트 트리거: 어깨가 충분히 내려갔다가 올라올 때
        if self.state == "UP" and shoulder > 50:  # 내려감 시작
            self.state = "DOWN"
        elif self.state == "DOWN" and shoulder <= 38:  # 다시 올라올 때 카운트
            self.state = "UP"
            if elbow >= 130:
                score, used, advice = self._snapshot_score(meta)
                return EvalResult(reps_delta=1, score=score, extra=used, advice=advice)
        return None

    # -------------------- 2. SHOULDER PRESS --------------------
    # 숫자 새는 기준: 팔이 올라갔다 내려올 때
    # 점수 시점: 팔이 위로 올라가 최대 높이일 때
    def _update_shoulder_press(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        shoulder = _avg_lr(meta, "shoulder")
        if shoulder is None:
            return None

        if self.state == "DOWN" and shoulder >= 150:  # 위로 올릴 때
            self.state = "UP"
            # 팔이 위로 최대치 → 점수
            score, used, advice = self._snapshot_score(meta)
            return EvalResult(reps_delta=1, score=score, extra=used, advice=advice)
        elif self.state == "UP" and shoulder <= 43:  # 내려갈 때
            self.state = "DOWN"
        return None

    # -------------------- 3. SIDE LATERAL RAISE --------------------
    # 숫자 새는 기준: 팔이 올라갔다 내려올 때
    # 점수 시점: 어깨가 43 이하일 때
    def _update_side_lateral(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        shoulder = _avg_lr(meta, "shoulder")
        if shoulder is None:
            return None

        if self.state == "DOWN" and shoulder >= 85:  # 위로 올림
            self.state = "UP"
        elif self.state == "UP" and shoulder <= 43:  # 내려오면서 카운트 + 점수
            self.state = "DOWN"
            score, used, advice = self._snapshot_score(meta)
            return EvalResult(reps_delta=1, score=score, extra=used, advice=advice)
        return None

    # -------------------- 4. DUMBBELL ROW --------------------
    # 숫자 새는 기준: 팔이 올라갔다 내려올 때
    # 점수 시점: 엘보우가 150도일 때
    def _update_dumbbell_row(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        elbow = _avg_lr(meta, "elbow")
        if elbow is None:
            return None

        if self.state == "DOWN" and elbow <= 125:  # 팔 당김
            self.state = "UP"
        elif self.state == "UP" and elbow >= 150:  # 팔 펴지며 복귀 → 점수
            self.state = "DOWN"
            score, used, advice = self._snapshot_score(meta)
            return EvalResult(reps_delta=1, score=score, extra=used, advice=advice)
        return None
