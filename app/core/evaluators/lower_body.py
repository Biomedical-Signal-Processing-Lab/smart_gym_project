# app/core/evaluators/lower_body.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
from PySide6.QtGui import QColor
from .base import ExerciseEvaluator, EvalResult

# ------------- 공용 유틸 -------------
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
        "hip":      [["hip_l","Hip(L)","hipL"], ["hip_r","Hip(R)","hipR"]],
        "knee":     [["knee_l","Knee(L)","kneeL"], ["knee_r","Knee(R)","kneeR"]],
        "shoulder": [["shoulder_l","Shoulder(L)"], ["shoulder_r","Shoulder(R)"]],
        "elbow":    [["elbow_l","Elbow(L)"], ["elbow_r","Elbow(R)"]],
        "hipline":  [["hipline_l","HipLine(L)"], ["hipline_r","HipLine(R)"]],
    }
    if base not in variants:
        return None
    l = _get_first(meta, variants[base][0])
    r = _get_first(meta, variants[base][1])
    if l is None and r is None: return None
    if l is None: return r
    if r is None: return l
    return (l + r) / 2.0


class LowerBodyEvaluator(ExerciseEvaluator):
    """
    하체 평가기: squat / leg_raise
    - 카운트 시점과 점수 시점을 분리
    - 점수 규칙: best=100, best↔max=80, best↔min=80, 그 외 50 (±3° 허용)
    """

    TOL = 3.0
    DEBOUNCE_N = 3

    # (기존 무릎 기반 동작 감지를 위해 둠)
    SQUAT_DOWN_TH = 120.0
    SQUAT_UP_TH   = 165.0

    # 레그 레이즈는 "팔(=어깨)이 올라갔다 내려올 때" 기준으로 전이 감지
    LR_SHOULDER_UP_TH   = 85.0
    LR_SHOULDER_DOWN_TH = 43.0

    THRESHOLDS: Dict[str, Dict[str, Dict[str, float]]] = {
        # 스쿼트: 이제 점수는 '어깨 + 팔꿈치' 평균으로 계산.
        # ↓↓↓ 필요하면 값만 바꿔도 됨(없으면 해당 항목은 스킵되고 무릎/힙으로 폴백)
        "squat": {
            # TODO: 현장 튜닝값을 넣어 사용. 임시로 범위를 넓게 잡아둠.
            "shoulder": {"best": 100.0, "max": 60.0, "min": 150.0},  # << 값 채워 사용 권장
            "elbow":    {"best": 100.0, "max": 60.0, "min": 150.0},  # << 값 채워 사용 권장
            # 폴백(임계 미설정 시 사용)
            "knee":     {"best": 100.0, "max": 60.0, "min": 150.0},
            "hip":      {"best": 100.0, "max": 65.0, "min": 150.0},
        },
        # 레그 레이즈: 점수는 '힙' 각도로 산출 (best=140, max=155, min=125)
        "leg_raise": {
            "hip": {"best": 140.0, "max": 155.0, "min": 125.0},
        },
    }

    def __init__(self, label: str = "squat"):
        super().__init__()
        self.mode = label
        self.reset()

    def reset(self) -> None:
        self.state = "UP"
        self._deb = 0

    # ---------- 공용 점수 ----------
    def _score_by_angle(self, angle: float, cfg: Dict[str, float]) -> int:
        b, mx, mn = cfg["best"], cfg["max"], cfg["min"]
        if abs(angle - b) <= self.TOL:
            return 100
        if mx <= angle < b:
            return 80
        if b < angle <= mn:
            return 80
        return 50

    # ---------- 스냅샷 점수 ----------
    def _score_snapshot(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        used: Dict[str, float] = {}
        scores: List[int] = []
        cfgs = self.THRESHOLDS[self.mode]

        if self.mode == "squat":
            # 우선순위: shoulder + elbow 평균, 둘 중 하나라도 임계치 없으면 무릎/힙으로 폴백
            sh = _avg_lr(meta, "shoulder")
            el = _avg_lr(meta, "elbow")
            if "shoulder" in cfgs and sh is not None:
                used["shoulder"] = sh
                scores.append(self._score_by_angle(sh, cfgs["shoulder"]))
            if "elbow" in cfgs and el is not None:
                used["elbow"] = el
                scores.append(self._score_by_angle(el, cfgs["elbow"]))

            # 둘 다 못 썼다면 기존 무릎/힙으로 폴백
            if not scores:
                kn = _avg_lr(meta, "knee")
                hp = _avg_lr(meta, "hip")
                if "knee" in cfgs and kn is not None:
                    used["knee"] = kn; scores.append(self._score_by_angle(kn, cfgs["knee"]))
                if "hip" in cfgs and hp is not None:
                    used["hip"] = hp; scores.append(self._score_by_angle(hp, cfgs["hip"]))

            advice = "무릎보다 상체(어깨/팔꿈치) 정렬을 유지하세요."

        else:  # leg_raise
            hip = _avg_lr(meta, "hip")
            if hip is not None:
                used["hip"] = hip
                scores.append(self._score_by_angle(hip, cfgs["hip"]))
            advice = "복부에 힘을 주고 천천히 내리세요."

        score = int(round(sum(scores) / max(1, len(scores))))
        return score, used, advice

    # ---------- 메인 업데이트 ----------
    def update_and_maybe_score(self, meta: Dict[str, Any], label: Optional[str] = None) -> Optional[EvalResult]:
        if label:
            self.mode = label

        if self.mode == "squat":
            return self._update_squat(meta)
        elif self.mode == "leg_raise":
            return self._update_leg_raise(meta)
        return None

    # ===== 스쿼트 =====
    # 숫자: 몸이 내려갔다가(무릎 각도 작아짐) 올라올 때(무릎 각도 커짐)
    # 점수: 같은 시점(UP 복귀 순간)에 스냅샷 평가

     # ✅ ExercisePage 표준 엔트리
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        if self.mode == "squat":
            return self._update_squat(meta)
        if self.mode == "leg_raise":
            return self._update_leg_raise(meta)
        return None

    def _update_squat(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        knee = _avg_lr(meta, "knee")
        if knee is None:
            self._deb = 0
            return None

        if self.state == "UP":
            if knee < self.SQUAT_DOWN_TH:
                self._deb += 1
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0
        else:  # DOWN
            if knee >= self.SQUAT_UP_TH:
                self._deb += 1
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    score, used, advice = self._score_snapshot(meta)  # ← 이 시점에 점수
                    return EvalResult(reps_delta=1, score=score, extra=used, advice=advice)
        return None

    # ===== 레그 레이즈 =====
    # 숫자: 팔(어깨)이 올라갔다(≥85) 내려올 때(≤43) 1회
    # 점수: 어깨가 43° 이하인 ‘그 순간’ 스냅샷(힙 각도 기준) 평가
    def _update_leg_raise(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        shoulder = _avg_lr(meta, "shoulder")
        if shoulder is None:
            self._deb = 0
            return None

        if self.state == "UP":
            # 팔 올리는 구간
            if shoulder >= self.LR_SHOULDER_UP_TH:
                self._deb += 1
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0
        else:  # DOWN 구간(내려오는 중)
            if shoulder <= self.LR_SHOULDER_DOWN_TH:
                self._deb += 1
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    score, used, advice = self._score_snapshot(meta)  # 힙 각도로 점수
                    return EvalResult(reps_delta=1, score=score, extra=used, advice=advice)
        return None
