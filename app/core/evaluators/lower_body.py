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

def _score_by(angle: Optional[float], best: float, maxv: float=None, minv: float=None, **kwargs) -> int:
    """minv~maxv 사이를 선형으로 0~100점 환산"""
    if maxv is None and "max" in kwargs:  maxv = kwargs["max"]
    if minv is None and "min" in kwargs:  minv = kwargs["min"]
    if angle is None or math.isnan(angle):
        return 0

    # minv < maxv 일 경우 swap
    if minv < maxv:
        minv, maxv = maxv, minv

    # 범위 바깥이면 0 또는 100으로 클램프
    if angle <= maxv: 
        return 0
    if angle >= minv:
        return 100

    # 선형 보간: maxv→0점, minv→100점
    ratio = (angle - maxv) / (minv - maxv)
    score = int(round(ratio * 100))
    return max(0, min(100, score))

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
    DEBOUNCE_N = 3

    # 스쿼트(무릎 전이 임계)
    SQUAT_DOWN_TH = 135.0
    SQUAT_UP_TH   = 165.0

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

    LEG_HIP = dict(best=140.0, max=165.0, min=110.0)

    def __init__(self, label: str = "squat"):
        assert label in ("squat", "leg_raise")
        self.mode = label
        super().__init__()

    def reset(self) -> None:
        self.state = "UP"
        self._deb = 0
        self._passed_up = False    
        self._cooldown = 0           
        self._last_emit_id = None     
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

        if hip is not None:
            sc_hip = _score_by(hip, **cfgs["hip"])
            used["hip"] = hip; scores.append(sc_hip)
           
        if knee is not None:
            sc_knee = _score_by(knee, **cfgs["knee"])
            used["knee"] = knee; scores.append(sc_knee)

        if not scores:
            return 50, used, "각도 인식 불가: 프레임/포즈 확인"

        score = int(round(sum(scores) / len(scores)))
        advice = "엉덩이·무릎 깊이를 함께 맞추세요. 90~100° 구간이 좋아요."
    
        return score, used, advice

    # ---------- 레그 레이즈 점수 ----------
    def _score_snapshot_legraise(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        used: Dict[str, float] = {}
        hip = _avg_lr(meta, "hip")
        used["hip"] = hip if hip is not None else float("nan")
        score = _score_by(hip, **self.LEG_HIP)   
        advice = "복부 긴장 유지, 천천히 내려오며 힙 각도 컨트롤!"
        _dbg(f"[LEG] 점수: hip={_fmt(hip)} → score={score}")
        return score, used, advice
    
    # ---------- 공개 API ----------
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        label = meta.get("label")
        
        _dbg(
                f"knee(L/R)={_fmt(meta.get('knee_l_deg'))}/{_fmt(meta.get('knee_r_deg'))}, "
                f"hip(L/R)={_fmt(meta.get('hip_l_deg'))}/{_fmt(meta.get('hip_r_deg'))}, "
                f"hipline(L/R)={_fmt(meta.get('hipline_l_deg'))}/{_fmt(meta.get('hipline_r_deg'))}"
            )

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
        hip  = _avg_lr(meta, "hip")

        # --- 유효성 검사 ---
        if knee is None or math.isnan(knee):
            self._deb = 0
            return None

        # ===== 상태 전환 로직 =====
        if self.state == "UP":
            # 내려가기 시작
            if knee < self.SQUAT_DOWN_TH:
                self._deb += 1
                
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"
                    self._deb = 0
                    # 최소각 초기화
                    self._min_knee = knee
                    self._min_hip  = hip
                    
        else:  # state == "DOWN"
            # 내려가는 동안 최소각 갱신
            if knee is not None:
                if not hasattr(self, "_min_knee") or self._min_knee is None or knee < self._min_knee:
                    self._min_knee = knee
            if hip is not None:
                if not hasattr(self, "_min_hip") or self._min_hip is None or hip < self._min_hip:
                    self._min_hip = hip

            _dbg(f"[SQUAT] DOWN tracking: min_knee={_fmt(self._min_knee)} min_hip={_fmt(self._min_hip)}")

            # 올라오기 감지
            if knee >= self.SQUAT_UP_TH:
                self._deb += 1
                
                if self._deb >= self.DEBOUNCE_N-1:
                    self.state = "UP"
                    self._deb = 0

                    # ====== 점수 계산: 최소각 기준 ======
                    meta2 = dict(meta)
                    if getattr(self, "_min_knee", None) is not None:
                        meta2["knee_l_deg"] = meta2["knee_r_deg"] = float(self._min_knee)
                    if getattr(self, "_min_hip", None) is not None:
                        meta2["hip_l_deg"] = meta2["hip_r_deg"] = float(self._min_hip)

                    score, used, advice = self._score_snapshot_squat(meta2)
                    
                    # 다음 세트 대비 초기화
                    self._min_knee = None
                    self._min_hip = None

                    return EvalResult(
                        rep_inc=1,
                        score=score,
                        advice=advice,
                        color=_color(score),
                    )

        return None


    # ---------- 레그 레이즈 ----------
    def _update_leg_raise(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        
        if getattr(self, "_cooldown", 0) > 0:
            self._cooldown -= 1
            return None
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

        _dbg(f"[LEG:R] tick hip_r={_fmt(hip_r)} state={self.state} deb={self._deb} passed_up={getattr(self,'_passed_up',False)}")

        # ====== 상태 전환 ======
        if self.state == "UP":
            # 밑으로 내려갈 때(각도 커짐) → DOWN 진입 후보
            if hip_r >= self.LR_DOWN_TH:
                self._deb += 1
                _dbg(f"[LEG:R] detect UP→DOWN trigger ({self._deb}/{self.DEBOUNCE_N})")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"
                    self._deb = 0
                    _dbg("[LEG:R] state→DOWN")

                    # ★ 직전에 위(UP)를 통과했다면, 지금 DOWN 도달이 '완주'이므로 점수+카운트
                    if getattr(self, "_passed_up", False):
                        # 1) 같은 프레임 중복 방지
                        frame_id = meta.get("frame_id") or meta.get("frame_idx") or meta.get("ts")
                        if frame_id is not None and frame_id == getattr(self, "_last_emit_id", None):
                            _dbg("[LEG:R] duplicate emit blocked by frame_id guard")
                            return None

                        score, used, advice = self._score_snapshot_legraise(meta)
                        _dbg(f"[LEG:R] COUNT at UP→DOWN (cycle complete) → score={score}")

                        # 2) 쿨다운/프레임ID 기록
                        self._last_emit_id = frame_id
                        self._cooldown = 5          # ← 필요시 3~8 사이로 조정

                        self._passed_up = False     # 다음 사이클 대비 리셋
                        return EvalResult(
                            rep_inc=1,
                            score=score,
                            advice=advice,
                            color=_color(score),
                        )

        else:  # state == "DOWN"
             
            if hip_r <= self.LR_UP_TH:
                self._deb += 1
                _dbg(f"[LEG:R] detect DOWN→UP trigger ({self._deb}/{self.DEBOUNCE_N})")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"
                    self._deb = 0
                    self._passed_up = True     
                    _dbg("[LEG:R] state→UP (passed_up=True)")

        return None
