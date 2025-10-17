from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
from PySide6.QtGui import QColor
from .base import ExerciseEvaluator, EvalResult

# ===== Debug helpers =====
DEBUG_LOWER = True  # 필요 시 False로 끄기

def _dbg(*args):
    if DEBUG_LOWER:
        print("[EVAL:lower]", *args)

def _fmt(v):
    if v is None: return "None"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

# -------- 공통 유틸 --------
def _avg_lr(meta: Dict[str, Any], base: str) -> Optional[float]:
    l = meta.get(f"{base}_l_deg"); r = meta.get(f"{base}_r_deg")
    if l is None and r is None: return None
    if l is None: return float(r)
    if r is None: return float(l)
    return (float(l) + float(r)) / 2.0

def _score_by(angle: float, best: float, maxv: float=None, minv: float=None, tol: float=3.0, **kwargs) -> int:
    # 'max'/'min' 키로 들어오는 경우 호환 처리
    if maxv is None and "max" in kwargs:  maxv = kwargs["max"]
    if minv is None and "min" in kwargs:  minv = kwargs["min"]

    _dbg(f"점수 계산 시작: angle={_fmt(angle)} | best={_fmt(best)} | max={_fmt(maxv)} | min={_fmt(minv)} | tol={tol}")

    if angle is None or (isinstance(angle, float) and math.isnan(angle)):
        _dbg(" -> 결과: INVALID (angle=None or NaN) → score=50")
        return 50
    if abs(angle - best) <= tol:
        _dbg(f" -> 결과: BEST 구간 (|angle-best|<={tol}) → score=100")
        return 100
    if maxv <= angle < best:
        _dbg(f" -> 결과: GOOD (max≤angle<best) → score=80")
        return 80
    if best < angle <= minv:
        _dbg(f" -> 결과: GOOD (best<angle≤min) → score=80")
        return 80
    _dbg(" -> 결과: OUT OF RANGE → score=50")
    return 50

def _color(s: int) -> QColor:
    if s >= 95:  return QColor(0, 200, 0)
    if s >= 80:  return QColor(255, 215, 0)
    return QColor(255, 80, 80)

_LABEL_ALIAS = {
    "squat": "squat",
    "스쿼트": "squat",
    "leg_raise": "leg_raise",
    "legraise": "leg_raise",
    "레그레이즈": "leg_raise",
}   

def _normalize_label(lbl: Any) -> Optional[str]:  # [ADD]
    if lbl is None: return None
    s = str(lbl).strip().lower().replace("-", "_").replace(" ", "")
    return _LABEL_ALIAS.get(s, s)


# -------- 파라미터(표 기준) --------
class LowerBodyEvaluator(ExerciseEvaluator):
    """
    ONLY: squat / leg_raise
    - 스쿼트: 카운트=무릎 DOWN→UP, 점수=힙+무릎 각도 평균(UP 복귀 스냅샷)
    - 레그 레이즈: 카운트=어깨 UP→(다시)DOWN, 점수=힙 각도(어깨 ≤ 43° 스냅샷)
    """

    # 공통
    TOL = 3.0
    DEBOUNCE_N = 2

    # 스쿼트(무릎 전이 임계)
    SQUAT_DOWN_TH = 120.0
    SQUAT_UP_TH   = 165.0

    # 레그 레이즈(어깨 전이/스냅샷 임계)
    LR_SHOULDER_UP_TH   = 120.0
    LR_SHOULDER_SNAP_TH = 45.0   # 스냅샷·카운트 타점

    LR_DOWN_TH = 155.0  # 다리 내릴 때 (hip_r_deg 커짐)
    LR_UP_TH   = 170.0  # 다리 올릴 때 (hip_r_deg 작아짐)

    # 점수 기준(표)
    THRESHOLDS: Dict[str, Dict[str, Dict[str, float]]] = {
        "squat": {
            "hip":  {"best": 100.0, "maxv": 65.0, "minv": 150.0},
            "knee": {"best": 100.0, "maxv": 60.0, "minv": 150.0},
        },
        "leg_raise": {
            "hip": {"best": 140.0, "max": 155.0, "min": 125.0},
        },
    }

    LEG_HIP = dict(best=140.0, max=155.0, min=125.0)

    def __init__(self, label: str = "squat"):
        assert label in ("squat", "leg_raise")
        self.mode = label
        super().__init__()

    def reset(self) -> None:
        self.state = "UP"
        self._deb = 0
        _dbg(f"reset() -> state=UP, deb=0, mode={getattr(self,'mode',None)}")

    # ---------- 점수 스냅샷 ----------
    def _score_snapshot(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        if self.mode == "squat":
            return self._score_snapshot_squat(meta)
        else:
            return self._score_snapshot_legraise(meta)

    # ---------- 스쿼트 점수 ----------
    def _score_snapshot_squat(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        used: Dict[str, float] = {}
        scores: List[int] = []
        cfgs = self.THRESHOLDS["squat"]

        hip  = _avg_lr(meta, "hip")
        knee = _avg_lr(meta, "knee")

        _dbg(f"[SQUAT] snapshot in: hip={_fmt(hip)} knee={_fmt(knee)}")

        if hip is not None:
            sc_hip = _score_by(hip, **cfgs["hip"])
            used["hip"] = hip; scores.append(sc_hip)
            _dbg(f"[SQUAT] hip score: angle={_fmt(hip)} → {sc_hip}")
        if knee is not None:
            sc_knee = _score_by(knee, **cfgs["knee"])
            used["knee"] = knee; scores.append(sc_knee)
            _dbg(f"[SQUAT] knee score: angle={_fmt(knee)} → {sc_knee}")

        if not scores:
            _dbg("[SQUAT] no angles → score=50")
            return 50, used, "각도 인식 불가: 프레임/포즈 확인"

        score = int(round(sum(scores) / len(scores)))
        advice = "엉덩이·무릎 깊이를 함께 맞추세요. 90~100° 구간이 좋아요."
        _dbg(f"[SQUAT] snapshot out: score={score} used={ {k:_fmt(v) for k,v in used.items()} }")
        return score, used, advice

    # ---------- 레그 레이즈 점수 ----------
    def _score_snapshot_legraise(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        used: Dict[str, float] = {}
        hip = _avg_lr(meta, "hip")
        used["hip"] = hip if hip is not None else float("nan")
        score = _score_by(hip, **self.LEG_HIP)  # [KEEP] max/min → 호환 처리로 OK
        advice = "복부 긴장 유지, 천천히 내려오며 힙 각도 컨트롤!"
        _dbg(f"[LEG] 점수: hip={_fmt(hip)} → score={score}")
        return score, used, advice
    
    # ---------- 공개 API ----------
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        label = meta.get("label")
        _dbg(f"[UPDATE] mode={self.mode} label={label} state={self.state} deb={self._deb}")

        if self.mode == "squat" and label != "squat":
            return None
        if self.mode == "leg_raise" and label != "leg_raise":
            return None

        if self.mode == "squat":
            return self._update_squat(meta)
        else:
            return self._update_leg_raise(meta)

    # ---------- 스쿼트 ----------
    def _update_squat(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        knee = _avg_lr(meta, "knee")
        if knee is None:
            self._deb = 0
            _dbg("임정민 : SQUAT knee=None -> skip")
            return None

        _dbg(f"임정민 :  SQUAT tick: knee={_fmt(knee)} state={self.state} deb={self._deb}")

        if self.state == "UP":
            if knee < self.SQUAT_DOWN_TH:
                self._deb += 1
                _dbg(f"SQUAT down_detect deb={self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0
                    _dbg("SQUAT state→DOWN")
        else:
            if knee >= self.SQUAT_UP_TH:
                self._deb += 1
                _dbg(f"SQUAT up_detect deb={self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    _dbg("SQUAT state→UP, call _score_snapshot_squat()")
                    score, used, advice = self._score_snapshot_squat(meta)
                    _dbg(f"임정민 :: SQUAT RESULT: rep_inc=1 score={score} used={ {k:_fmt(v) for k,v in used.items()} }")
                    return EvalResult(
                        rep_inc=1,
                        score=score,
                        advice=advice,
                        color=_color(score),
                        # extra=used
                    )
        return None

    # ---------- 레그 레이즈 ----------
    def _update_leg_raise(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
    # 오른쪽 다리 각도(hip_r_deg)를 기준으로 감지
        hip_r = meta.get("hip_r_deg")
        try:
            hip_r = float(hip_r) if hip_r is not None else None
        except Exception:
            hip_r = None

        # NaN/None 방어
        if hip_r is None or (isinstance(hip_r, float) and math.isnan(hip_r)):
            self._deb = 0
            _dbg(f"[LEG:R] tick hip_r=None state={self.state} deb={self._deb}")
            return None

        _dbg(f"[LEG:R] tick hip_r={_fmt(hip_r)} state={self.state} deb={self._deb}")

        # ====== 상태 전환 ======
        if self.state == "UP":
            # 다리를 내릴 때 (엉덩이 각도 커짐)
            if hip_r >= self.LR_DOWN_TH:
                self._deb += 1
                _dbg(f"[LEG:R] detect UP→DOWN trigger ({self._deb}/{self.DEBOUNCE_N})")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"
                    self._deb = 0
                    _dbg("[LEG:R] state→DOWN")
        else:
            # 다리를 들어올릴 때 (엉덩이 각도 작아짐)
            if hip_r <= self.LR_UP_TH:
                self._deb += 1
                _dbg(f"[LEG:R] detect DOWN→UP trigger ({self._deb}/{self.DEBOUNCE_N})")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"
                    self._deb = 0
                    _dbg("[LEG:R] state→UP, call _score_snapshot_legraise()")
                    score, used, advice = self._score_snapshot_legraise(meta)
                    _dbg(f"[LEG:R] 점수는: rep_inc=1 score={score}")
                    return EvalResult(
                        rep_inc=1,
                        score=score,
                        advice=advice,
                        color=_color(score),
                    )

        return None
