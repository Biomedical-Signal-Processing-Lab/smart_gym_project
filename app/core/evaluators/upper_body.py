# app/core/evaluators/upper_body.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from PySide6.QtGui import QColor
from .base import ExerciseEvaluator, EvalResult
from .advice import get_advice

# =================== Debug & helpers ===================
DEBUG_UPPER = True
def _dbg(*args):
    if DEBUG_UPPER:
        print("[EVAL:upper]", *args)

def _fmt(v):
    if v is None: return "None"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)

def _color(s: int) -> QColor:
    if s is None: return QColor(200, 200, 200)
    if s >= 95:  return QColor(0, 200, 0)
    if s >= 80:  return QColor(255, 215, 0)
    return QColor(255, 80, 80)

# 라벨 정규화 (대소문자/공백/대시/한글 대응)
_LABEL_ALIAS = {
    "shoulder_press": "shoulder_press", "shoulderpress": "shoulder_press", "숄더프레스": "shoulder_press",
    "side_lateral_raise": "side_lateral_raise", "sidelateralraise": "side_lateral_raise",
    "side_lateral": "side_lateral_raise", "side_lateral-raise": "side_lateral_raise",
    "dumbbell_row": "dumbbell_row", "dumbbellrow": "dumbbell_row", "덤벨로우": "dumbbell_row",
    # 흔히 오는 원본 라벨들
    "side_lateral_raise(r)": "side_lateral_raise",
    "side_lateral_raise(l)": "side_lateral_raise",
    "side_lateral_raise_r": "side_lateral_raise",
    "side_lateral_raise_l": "side_lateral_raise",
    "dumbbell_row(r)": "dumbbell_row",
    "dumbbell_row(l)": "dumbbell_row",
}
def _normalize_label(lbl: Optional[str]) -> Optional[str]:
    if not lbl: return None
    s = str(lbl).strip().lower().replace("-", "_").replace(" ", "")
    return _LABEL_ALIAS.get(s, s)

# -------------------- meta 동기화: 유효한 값만 (표기용+snake 동시 기록) --------------------
SNAKE_MAP = {
    "Knee(L)": "knee_l_deg",       "Knee(R)": "knee_r_deg",
    "Hip(L)": "hip_l_deg",         "Hip(R)": "hip_r_deg",
    "Shoulder(L)": "shoulder_l_deg","Shoulder(R)": "shoulder_r_deg",
    "Elbow(L)": "elbow_l_deg",     "Elbow(R)": "elbow_r_deg",
    "HipLine(L)": "hipline_l_deg", "HipLine(R)": "hipline_r_deg",
    "Knee": "knee_avg_deg",        "Hip": "hip_avg_deg",
    "Shoulder": "shoulder_avg_deg","Elbow": "elbow_avg_deg",
    "HipLine": "hipline_avg_deg",
}
def _put_angle(meta: Dict[str, Any], display_key: str, value: Optional[float]) -> None:
    """표기용 키와 snake_case 키를 동시에 써준다 (value가 유효할 때만)"""
    if value is None:
        return
    try:
        v = float(value)
    except Exception:
        return
    meta[display_key] = v
    sk = SNAKE_MAP.get(display_key)
    if sk:
        meta[sk] = v

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
    # 표기키 + *_deg 모두 지원
    cand_l = [f"{base}_l", f"{base}_L", f"{base}(L)", base.capitalize()+"(L)", f"{base}_l_deg"]
    cand_r = [f"{base}_r", f"{base}_R", f"{base}(R)", base.capitalize()+"(R)", f"{base}_r_deg"]
    l = _get_first(meta, cand_l)
    r = _get_first(meta, cand_r)
    if l is None and r is None: return None
    if l is None: return r
    if r is None: return l
    return (l + r) / 2.0

def _right_only(meta: Dict[str, Any], base: str) -> Optional[float]:
    """오른쪽 관절만 사용 (표기/스네이크 혼용 대비)"""
    return _get_first(meta, [f"{base}_r", f"{base}_R", f"{base}(R)", base.capitalize()+"(R)", f"{base}_r_deg"])

# -------------------- UpperBodyEvaluator --------------------
class UpperBodyEvaluator(ExerciseEvaluator):
    
    # ----------공통 ----------
    DEBOUNCE_N = 2
    TOL = 3.0

    # 각도 기준 테이블 (lower와 동일 컨벤션 허용: max/min 또는 maxv/minv)
    THRESHOLDS = {
        
        "shoulder_press": {
            "elbow": {"best": 100.0, "max": 160.0, "min": 80.0},
        },
        "side_lateral_raise": {
            "shoulder": {"best":60.0, "max": 80.0, "min": 40.0},
           # "elbow":    {"best": 176.5, "max": 180.0, "min": 130.0},
        },
        "dumbbell_row": {
            "shoulder": {"best": 30.0,  "max": 45.0, "min": 18.0},
            "elbow":    {"best": 120.0, "max": 165.0, "min": 145.0},
        },
    }

    def __init__(self, label: str):
        
        norm = _normalize_label(label)
        self.mode = {"Side_lateral_raise":"side_lateral_raise",
                     "Dumbbell_Row":"dumbbell_row"}.get(label, norm)
        super().__init__()

    def reset(self):
        if getattr(self, "mode", None) == "shoulder_press":
            self.state = "DOWN"
        else:
            self.state = "UP"
    
        self._deb = 0
        self._cooldown = 0
        self._ema_sh = None    
        self._ema_hl = None    
        self._armed = False
        self._top_el = None

    
    def _snapshot_score(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
            used: Dict[str, float] = {}
            scores: List[int] = []
            cfgs = self.THRESHOLDS[self.mode]

            TILT_LIMIT   = 5.0
            LUMBAR_LIMIT = 10.0
            torso_jitter = float(meta.get("torso_jitter", 0.0))
            lumbar_ext   = float(meta.get("lumbar_ext", 0.0))
            elbow_flare  = bool(meta.get("elbow_flare", False))


            # 운동별 로직 구분
            if self.mode == "shoulder_press":
                e = _avg_lr(meta, "elbow")
                _put_angle(meta, "Elbow", e)
                if e is not None:
                    scores.append(self._score_shoulder_press(e, cfgs["elbow"]))  # ← 팔꿈치만
                    used["elbow"] = e
                    ctx = {
                            "elbow_flare": elbow_flare,
                            "tilt_instability": torso_jitter > TILT_LIMIT,
                            "back_arch":        lumbar_ext   > LUMBAR_LIMIT,
                            # shoulder press에서 팔꿈치 '깊이' 부족 감지(너무 큰 각도)시 힌트
                            "elbow_not_deep":   (e is not None and e > 120.0),
                        }

            elif self.mode == "side_lateral_raise":
                s = _avg_lr(meta, "shoulder")
                _put_angle(meta, "Shoulder", s)
                scores.clear()  # 안전
                if s is not None:
                    # 어깨 각도만 중앙형(삼각형)으로 점수
                    scores.append(self._score_side_lateral(s, cfgs["shoulder"]))
                used = {"shoulder": s if s is not None else float("nan")}
                #advice = "팔을 몸통과 수평 근처까지만 부드럽게 들어 올리고, 내려올 때 컨트롤하세요."
                ctx = {
                           "elbow_flare": elbow_flare,
                            "tilt_instability": torso_jitter > TILT_LIMIT,
                            "back_arch":        lumbar_ext   > LUMBAR_LIMIT,
                        }
                
            elif self.mode == "dumbbell_row":
                s = _avg_lr(meta, "shoulder")
                e = _avg_lr(meta, "elbow")
                _put_angle(meta, "Shoulder", s)
                _put_angle(meta, "Elbow", e)
                if s: scores.append(self._score_dumbbell_row(s, cfgs["shoulder"]))
                if e: scores.append(self._score_dumbbell_row(e, cfgs["elbow"]))
                ctx = {
                        "tilt_instability": torso_jitter > TILT_LIMIT,
                        "back_arch":        lumbar_ext   > LUMBAR_LIMIT,
                    }

            else:
                return 50, {}, "지원되지 않는 운동입니다."

            # 평균 점수 산출
            score = int(round(sum(scores) / max(1, len(scores))))
            advice_text = get_advice(_advice_key(self.mode), score, ctx)


            return score, used, advice


    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        m = self.mode
        
        if m == "shoulder_press":
            return self._update_shoulder_press(meta)
        if m == "side_lateral_raise":           
            return self._update_side_lateral(meta)
        if m == "dumbbell_row":
            return self._update_dumbbell_row(meta)
        return None

    def update_and_maybe_score(self, meta: Dict[str, Any], label: Optional[str] = None) -> Optional[EvalResult]:
            if label:
                self.mode = _normalize_label(label)
            return self.update(meta)

      

    # -------------------- Shoulder Press --------------------

    def _update_shoulder_press(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
       
        
        cfg = self.THRESHOLDS["shoulder_press"]["elbow"]
        el_min, el_best = float(cfg["min"]), float(cfg["best"])   
     
        EL_UP_TH   = 125                      
        EL_DOWN_TH = 110
        DEB_N      = 2               
        COOLDOWN_N = 6

        
        el = _avg_lr(meta, "elbow")
        if el is None:
            self._deb = 0
            return None
        _put_angle(meta, "Elbow", el)
        if el is None:
            self._deb = 0
            return None

        # ---- 쿨다운 ----
        if not hasattr(self, "_cooldown"): self._cooldown = 0
        if self._cooldown > 0: self._cooldown -= 1

        
        def bump(cond: bool):
            if cond:
                self._deb = min(self._deb + 1, DEB_N)
            else:
               
                self._deb = max(self._deb - 1, 0)

        if self.state == "DOWN":
            bump(el >= EL_UP_TH)
            if self._deb >= DEB_N and self._cooldown == 0:
                self.state = "UP"
                self._deb = 0
                
                self._top_el = el
                
        else:  # state == "UP"
           
            if el is not None:
                self._top_el = max(self._top_el or el, el)

            bump(el <= EL_DOWN_TH)
            if self._deb >= DEB_N:
                self.state = "DOWN"
                self._deb = 0
                self._cooldown = COOLDOWN_N

              
                meta2 = dict(meta)
                if getattr(self, "_top_el", None) is not None:
                    meta2["elbow_l_deg"] = meta2["elbow_r_deg"] = float(self._top_el)

                score, used, advice = self._snapshot_score(meta2)
                self._top_el = None  

                return EvalResult(
                    rep_inc=1,
                    score=score,
                    advice=advice,
                    color=_color(score),
                )

        return None

    
    def _score_shoulder_press(self, angle: Optional[float], cfg: Dict[str, float]) -> int:
     
        if angle is None:
            return 0
        
        best = cfg.get("best", 100.0)
        mn = cfg.get("min", best - 20.0)
        mx = cfg.get("max", best + 20.0)
        
       
        if angle <= mn or angle >= mx:
            return 0
        
       
        span = (mx - mn) / 2.0
        dist = abs(angle - best)
        score = 100 * (1 - dist / span)
        
        return max(0, min(100, int(round(score))))
    
    # -------------------- side_lateral--------------------

    def _update_side_lateral(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        
        SH_UP_TH   = 80.0   # 팔 올림 인식
        SH_DOWN_TH = 60.0   # 팔 내림 인식
        DEB_N      = self.DEBOUNCE_N
        COOLDOWN_N = 4

        s = _avg_lr(meta, "shoulder")
    
        if s is None:
            self._deb = 0
            return None
        _put_angle(meta, "Shoulder", s)

        if not hasattr(self, "_cooldown"): self._cooldown = 0
        if self._cooldown > 0: self._cooldown -= 1

        # 끈적 디바운스(옵션): 조건 만족 시 +1, 불만족 시 0으로 리셋 (심플 버전)
        if self.state == "DOWN":
            if s >= SH_UP_TH:
                self._deb += 1
                if self._deb >= DEB_N and self._cooldown == 0:
                    self.state = "UP"
                    self._deb = 0
            else:
                self._deb = 0
        else:  # state == "UP"
            if s <= SH_DOWN_TH:
                self._deb += 1
                if self._deb >= DEB_N:
                    self.state = "DOWN"
                    self._deb = 0
                    self._cooldown = COOLDOWN_N

                    # ↓ 지금 내려온 '현재 어깨 각도'로 점수 계산 (중앙형)
                    meta2 = dict(meta)
                    meta2["shoulder_l_deg"] = meta2["shoulder_r_deg"] = float(s)

                    score, used, advice = self._snapshot_score(meta2)
                    return EvalResult(
                        rep_inc=1,
                        score=score,
                        advice=advice,
                        color=_color(score),
                    )
            else:
                self._deb = 0

        return None

    def _score_side_lateral(self, angle: Optional[float], cfg: Dict[str, float]) -> int:
        
        if angle is None:
            return 0
        
        best = cfg.get("best", 60.0)
        mn = cfg.get("min", best - 20.0)
        mx = cfg.get("max", best + 20.0)
        
        # 범위 밖이면 0점
        if angle <= mn or angle >= mx:
            return 0
        
        # 중앙형(삼각형) 스코어 계산
        span = (mx - mn) / 2.0
        dist = abs(angle - best)
        score = 100 * (1 - dist / span)
        
        return max(0, min(100, int(round(score))))
    


    # -------------------- 4. DUMBBELL ROW --------------------

    def _update_dumbbell_row(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        elbow = _avg_lr(meta, "elbow")
        if elbow is None: self._deb = 0; return None

        if self.state == "UP":                    
            if elbow <= 145.0:
                self._deb += 1; 
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0; 
            else:
                self._deb = 0
        else:                                     
            if elbow >= 165.0:
                self._deb += 1
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    score, used, advice = self._snapshot_score(meta)
                    return EvalResult(rep_inc=1, score=score, advice=advice, color=_color(score))
            else:
                self._deb = 0
        return None


    def _score_dumbbell_row(self, angle: Optional[float], cfg: Dict[str, float]) -> int:
        """덤벨로우: 팔꿈치 145~165° 구간 유지 시 높은 점수 (plateau형)"""
        if angle is None:
            return 50
        mn, mx = cfg["min"], cfg["max"]
        if angle < mn:
            return 0
        if mn <= angle <= mx:
            # plateau zone (165 근처면 최고점)
            return int(80 + 20 * (angle - mn) / (mx - mn))
        return 100 if angle > mx else 0

   