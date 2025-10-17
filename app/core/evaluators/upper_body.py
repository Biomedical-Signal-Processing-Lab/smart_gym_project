# app/core/evaluators/upper_body.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from PySide6.QtGui import QColor
from .base import ExerciseEvaluator, EvalResult

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
    "pushup": "pushup", "푸쉬업": "pushup",
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
    variants = {
        "shoulder": [["shoulder_l","Shoulder(L)"], ["shoulder_r","Shoulder(R)"]],
        "elbow":    [["elbow_l","Elbow(L)"], ["elbow_r","Elbow(R)"]],
        "knee":     [["knee_l","Knee(L)"], ["knee_r","Knee(R)"]],
        "hip":      [["hip_l","Hip(L)"],   ["hip_r","Hip(R)"]],  # 상체에선 잘 안쓰지만 통일을 위해 추가
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
    """오른쪽 관절만 사용 (표기/스네이크 혼용 대비)"""
    return _get_first(meta, [f"{base}_r", f"{base}_R", f"{base}(R)", base.capitalize()+"(R)", f"{base}_r_deg"])

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

    # 각도 기준 테이블 (lower와 동일 컨벤션 허용: max/min 또는 maxv/minv)
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
        # mode를 먼저 세팅 (부모 __init__에서 reset() 호출 전에 준비)
        norm = _normalize_label(label)
        self.mode = {"Side_lateral_raise":"side_lateral_raise",
                     "Dumbbell_Row":"dumbbell_row"}.get(label, norm)
        super().__init__()

    def reset(self):
        self.state = "UP"
        self._deb = 0
        _dbg(f"reset() mode={getattr(self,'mode', None)}")

    # ---------- 점수 계산 유틸 (best/max/min 이름 통일 허용) ----------
    def _score_by_angle(self, angle: Optional[float], cfg: Dict[str, float]) -> int:
        if angle is None:
            _dbg("  - angle=None → score=50")
            return 50
        b = cfg.get("best")
        mx = cfg.get("maxv", cfg.get("max"))
        mn = cfg.get("minv", cfg.get("min"))
        _dbg(f"  - score_by: angle={_fmt(angle)} best={_fmt(b)} max={_fmt(mx)} min={_fmt(mn)} tol={self.TOL}")
        if b is None or mx is None or mn is None:
            _dbg("  - cfg invalid → score=50")
            return 50
        if abs(angle - b) <= self.TOL:
            return 100
        if mx <= angle < b:
            return 80
        if b < angle <= mn:
            return 80
        return 50

    # ---------- 스냅샷 점수 ----------
    def _snapshot_score(self, meta: Dict[str, Any]) -> Tuple[int, Dict[str, float], str]:
        """현재 프레임에서 관련 관절 평균점수 계산 + meta 동기화(표기용/스네이크)"""
        used: Dict[str, float] = {}
        scores: List[int] = []
        cfgs = self.THRESHOLDS[self.mode]

        if self.mode == "pushup":
            s = _right_only(meta, "shoulder")
            e = _right_only(meta, "elbow")
            k = _right_only(meta, "knee")
            _put_angle(meta, "Shoulder(R)", s)
            _put_angle(meta, "Elbow(R)", e)
            _put_angle(meta, "Knee(R)", k)
            if s is not None:
                sc = self._score_by_angle(s, cfgs["shoulder_r"]); scores.append(sc); used["shoulder_r"] = s
                _dbg(f"  - PUSHUP shoulder_r={_fmt(s)} → {sc}")
            if e is not None:
                sc = self._score_by_angle(e, cfgs["elbow_r"]); scores.append(sc); used["elbow_r"] = e
                _dbg(f"  - PUSHUP elbow_r={_fmt(e)} → {sc}")
            if k is not None:
                sc = self._score_by_angle(k, cfgs["knee_r"]); scores.append(sc); used["knee_r"] = k
                _dbg(f"  - PUSHUP knee_r={_fmt(k)} → {sc}")
            advice = "팔과 몸의 각도를 일정하게 유지하세요."

        elif self.mode == "shoulder_press":
            s = _avg_lr(meta, "shoulder"); e = _avg_lr(meta, "elbow")
            _put_angle(meta, "Shoulder", s)
            _put_angle(meta, "Elbow", e)
            if s is not None:
                sc = self._score_by_angle(s, cfgs["shoulder"]); scores.append(sc); used["shoulder"] = s
                _dbg(f"  - SP shoulder={_fmt(s)} → {sc}")
            if e is not None:
                sc = self._score_by_angle(e, cfgs["elbow"]); scores.append(sc); used["elbow"] = e
                _dbg(f"  - SP elbow={_fmt(e)} → {sc}")
            advice = "팔을 천천히 위로 올리세요."

        elif self.mode == "side_lateral_raise":
            s = _avg_lr(meta, "shoulder"); e = _avg_lr(meta, "elbow")
            _put_angle(meta, "Shoulder", s)
            _put_angle(meta, "Elbow", e)
            if s is not None:
                sc = self._score_by_angle(s, cfgs["shoulder"]); scores.append(sc); used["shoulder"] = s
                _dbg(f"  - SLR shoulder={_fmt(s)} → {sc}")
            if e is not None:
                sc = self._score_by_angle(e, cfgs["elbow"]); scores.append(sc); used["elbow"] = e
                _dbg(f"  - SLR elbow={_fmt(e)} → {sc}")
            advice = "팔이 너무 높지 않게 들어주세요."

        elif self.mode == "dumbbell_row":
            s = _avg_lr(meta, "shoulder"); e = _avg_lr(meta, "elbow")
            _put_angle(meta, "Shoulder", s)
            _put_angle(meta, "Elbow", e)
            if s is not None:
                sc = self._score_by_angle(s, cfgs["shoulder"]); scores.append(sc); used["shoulder"] = s
                _dbg(f"  - ROW shoulder={_fmt(s)} → {sc}")
            if e is not None:
                sc = self._score_by_angle(e, cfgs["elbow"]); scores.append(sc); used["elbow"] = e
                _dbg(f"  - ROW elbow={_fmt(e)} → {sc}")
            advice = "등 근육으로 팔을 당기세요."

        score = int(round(sum(scores) / max(1, len(scores))))
        _dbg(f"[SNAPSHOT] mode={self.mode} score={score} used={ {k:_fmt(v) for k,v in used.items()} }")
        return score, used, advice

    # ==================== 공개 API ====================
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        m = self.mode
        _dbg(f"[UPDATE] mode={m} state={self.state} deb={self._deb}")
        if m == "pushup":
            return self._update_pushup(meta)
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

    # -------------------- 1. PUSH-UP --------------------
# 전이: 어깨_r 기준 (DOWN: >= 45, UP: <= 38)  — 필요시 숫자만 조정
# 점수: UP 복귀 순간, "오른쪽 엘보우 >= 130" 이면 스냅샷
    def _update_pushup(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        shoulder = _right(meta, "shoulder")   # <- _right_only 말고 _right 사용
        elbow    = _right(meta, "elbow")

        _dbg(f"[PU] tick sh_r={_fmt(shoulder)} el_r={_fmt(elbow)} state={self.state} deb={self._deb}")

        # 전이는 '어깨'만으로 진행 (엘보우 None이어도 카운트·상태전이 가능)
        if shoulder is None:
            self._deb = 0
            _dbg("[PU] shoulder=None → skip tick (deb reset)")
            return None

        if self.state == "UP":
            # DOWN 진입 탐지 (내려감 시작)
            if shoulder >= 38.0:
                self._deb += 1
                _dbg(f"[PU] DOWN detect deb={self._deb}/{self.DEBOUNCE_N} (sh>={shoulder:.1f})")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0
                    _dbg("[PU] state→DOWN")
            else:
                # 조건 벗어나면 디바운스 리셋
                if self._deb:
                    _dbg("[PU] DOWN detect canceled → deb=0")
                self._deb = 0

        else:  # state == "DOWN"
            # UP 복귀 탐지
            if shoulder <= 45.0:
                self._deb += 1
                _dbg(f"[PU] UP detect deb={self._deb}/{self.DEBOUNCE_N} (sh<={shoulder:.1f})")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    _dbg("[PU] state→UP (snapshot try)")

                    # 스냅샷/점수: 엘보우 조건 확인
                    if elbow is not None and elbow >= 130.0:
                        score, used, advice = self._score_snapshot_pushup(meta)  # ← 함수명 확인!
                        _dbg(f"[PU] RESULT: rep_inc=1 score={score} used={ {k:_fmt(v) for k,v in used.items()} }")
                        return EvalResult(
                            rep_inc=1,
                            score=score,
                            advice=advice,
                            color=_color(score),
                        )
                    else:
                        _dbg(f"[PU] snapshot skipped (elbow<130 or None): el={_fmt(elbow)}")
                        # 엘보우 조건 미충족이면 '카운트는 하지 않음' — 필요 시 정책 변경 가능
            else:
                if self._deb:
                    _dbg("[PU] UP detect canceled → deb=0")
                self._deb = 0

        return None

    # -------------------- 2. SHOULDER PRESS --------------------
    # 카운트: 팔이 올라갔다 내려올 때
    # 점수: 팔이 위로 최대치(>=150)일 때
    def _update_shoulder_press(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        shoulder = _avg_lr(meta, "shoulder")
        _dbg(f"[SP] tick shoulder_avg={_fmt(shoulder)} state={self.state} deb={self._deb}")
        if shoulder is None:
            return None

        if self.state == "DOWN":
            if shoulder >= 43:
                self._deb += 1
                _dbg(f"[SP] up_detect {self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    score, used, advice = self._snapshot_score(meta)
                    _dbg(f"[SP] RESULT rep_inc=1 score={score}")
                    return EvalResult(
                        rep_inc=1,
                        score=score,
                        advice=advice,
                        color=_color(score),
                    )
        else:  # UP
            if shoulder <= 150:
                self._deb += 1
                _dbg(f"[SP] down_detect {self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0
                    _dbg("[SP] state→DOWN")
        return None

    # -------------------- 3. SIDE LATERAL RAISE --------------------
    # 카운트: 팔이 올라갔다 내려올 때
    # 점수: 어깨가 43 이하일 때(내려오면서)
    def _update_side_lateral(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        shoulder = _avg_lr(meta, "shoulder")
        _dbg(f"[SLR] tick shoulder_avg={_fmt(shoulder)} state={self.state} deb={self._deb}")
        if shoulder is None:
            return None

        if self.state == "DOWN":
            if shoulder >= 85:
                self._deb += 1
                _dbg(f"[SLR] up_detect {self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    _dbg("[SLR] state→UP")
        else:  # UP
            if shoulder <= 43:
                self._deb += 1
                _dbg(f"[SLR] down_detect {self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0
                    score, used, advice = self._snapshot_score(meta)
                    _dbg(f"[SLR] RESULT rep_inc=1 score={score}")
                    return EvalResult(
                        rep_inc=1,
                        score=score,
                        advice=advice,
                        color=_color(score),
                    )
        return None

    # -------------------- 4. DUMBBELL ROW --------------------
    # 카운트: 팔이 당겨졌다가(≤125) 다시 펴질 때(≥150)
    # 점수: 팔 펴지며 복귀 시점
    def _update_dumbbell_row(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        elbow = _avg_lr(meta, "elbow")
        _dbg(f"[ROW] tick elbow_avg={_fmt(elbow)} state={self.state} deb={self._deb}")
        if elbow is None: self._deb = 0; return None

        if self.state == "UP":                     # 팔 펴진 상태 → 당기면 DOWN
            if elbow <= 145.0:
                self._deb += 1; _dbg(f"[ROW] to DOWN {self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "DOWN"; self._deb = 0; _dbg("[ROW] state→DOWN")
            else:
                self._deb = 0
        else:                                      # 수축 → 펴지면 스냅샷
            if elbow >= 165.0:
                self._deb += 1; _dbg(f"[ROW] to UP(snap) {self._deb}/{self.DEBOUNCE_N}")
                if self._deb >= self.DEBOUNCE_N:
                    self.state = "UP"; self._deb = 0
                    score, used, advice = self._snapshot_score(meta)
                    _dbg(f"[ROW] RESULT rep_inc=1 score={score}")
                    return EvalResult(rep_inc=1, score=score, advice=advice, color=_color(score))
            else:
                self._deb = 0
        return None
