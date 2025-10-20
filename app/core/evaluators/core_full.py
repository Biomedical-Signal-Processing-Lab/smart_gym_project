from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
from PySide6.QtGui import QColor
from .base import ExerciseEvaluator, EvalResult
import math
try:
    from .advice import get_advice
except Exception:
    from advice import get_advice

# ===== Debug helpers =====
DEBUG_LOWER = False  # 필요 시 False로 끄기

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


# -------- 파라미터(표 기준) --------
class CoreBodyEvaluator(ExerciseEvaluator):
    """
    ONLY: squat / leg_raise
    - 스쿼트: 카운트=무릎 DOWN→UP, 점수=힙+무릎 각도 평균(UP 복귀 스냅샷)
    - 레그 레이즈: 카운트=어깨 UP→(다시)DOWN, 점수=힙 각도(어깨 ≤ 43° 스냅샷)
    """

    def __init__(self, label: str = "pushup"):
        assert label in ("burpee", "pushup", "jumping_jack")
        self.mode = label
        super().__init__()
        self.complit = False
        self.do = False

        # pushup 상태
        self._pu_state = "UP"      # "UP" 또는 "DOWN"
        self._pu_min_elbow = None  # 한 rep 동안 가장 작은 팔꿈치각
        self._pu_min_knee  = None  # 한 rep 동안 가장 작은 무릎각(굽힌 순간)

    def reset(self) -> None:
        # 공통 상태
        self.prev_label = getattr(self, "mode", None)

        # 디바운스/상태
        self.state = "UP"        # 필요 없으면 삭제해도 됨 (호환용)
        self._deb = 0

        # pushup용 상태
        self._pu_state = "UP"
        self._pu_min_elbow = None
        self._pu_min_knee  = None

        _dbg(f"reset() mode={getattr(self,'mode', None)}")
    # ---------- 공개 API ----------


    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:

        # 외부 meta['label']로 필터링하지 말 것
        self.prev_label = getattr(self, "mode", None)
        m = self.mode  # "burpee" | "pushup" | "Jumping_jack"

        if m == "burpee":
            res = self._update_burpee(meta)
            return res if res is not None else EvalResult()

        if m == "pushup":
            res = self._update_pushup(meta)
            return res if res is not None else EvalResult()

        if m == "jumping_jack":  # 네가 쓰는 케이스 그대로 유지
            res = self._update_Jumping_jack(meta)
            return res if res is not None else EvalResult()

        # 알 수 없는 모드일 때 빈 결과
        return EvalResult()


        # label = meta.get("label")
        
        # _dbg(
        #         f"knee(L/R)={_fmt(meta.get('knee_l_deg'))}/{_fmt(meta.get('knee_r_deg'))}, "
        #         f"hip(L/R)={_fmt(meta.get('hip_l_deg'))}/{_fmt(meta.get('hip_r_deg'))}, "
        #         f"shoulder(L/R)={_fmt(meta.get('shoulder_l_deg'))}/{_fmt(meta.get('shoulder_r_deg'))}, "
        #         f"elbow(L/R)={_fmt(meta.get('elbow_l_deg'))}/{_fmt(meta.get('elbow_r_deg'))}, "
        #         f"hipline(L/R)={_fmt(meta.get('hipline_l_deg'))}/{_fmt(meta.get('hipline_r_deg'))}"
        #     )
        
        # self.prev_label = label

        # if self.mode == "burpee" and label != "burpee":
        #     return None
        # if self.mode == "pushup" and label != "pushup":
        #     return None
        # if self.mode == "Jumping_jack" and label != "Jumping_jack":
        #     return None
        


        

        # if self.mode == "burpee":
        #     return self._update_burpee(meta)
        
        # if self.mode == "pushup":
        #     return self._update_pushup(meta)
        # else:
        #     return self._update_Jumping_jack(meta)
        

    # ---------- burpee ----------
    def _update_burpee(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        # ---- 오른쪽 어깨각만 사용 ----
        s = meta.get("shoulder_r_deg")
        try:
            s = float(s)
        except Exception:
            self.prev_label = "burpee"; return None
        if not math.isfinite(s):
            self.prev_label = "burpee"; return None

        # ---- 문턱(덜 보수적) ----
        DOWN_ENTER = 110.0   # <= 이하면 Down 인정 (프라이밍)
        UP_ENTER   = 110.0   # >= 이면 Up 인정 (카운트 트리거)

        # ---- 상태: Down을 먼저 지나야 Up에서 카운트 ----
        state  = getattr(self, "_bp_state", "EXPECT_DOWN")  # "EXPECT_DOWN" or "EXPECT_UP"
        prev_s = getattr(self, "_bp_prev_s", s)

        # 1) Down 프라이밍(한 번 내려가야 함)
        if state == "EXPECT_DOWN":
            if (prev_s > DOWN_ENTER) and (s <= DOWN_ENTER):
                state = "EXPECT_UP"
            self._bp_state = state
            self._bp_prev_s = s
            self.prev_label = "burpee"
            return None

        # 2) Up에서 카운트 (Down을 거쳤을 때만)
        #    엣지: prev_s < UP_ENTER <= s
        if (prev_s < UP_ENTER) and (s >= UP_ENTER):
            # 카운트 즉시 + 상태 리암
            self._bp_state = "EXPECT_DOWN"
            self._bp_prev_s = s
            self.prev_label = "burpee"
            score = 100
            advice_text = get_advice("jumping_jack", score, ctx=None)
            return EvalResult(
                rep_inc=1,
                score=100,                 # 간단히 고정 점수
                advice=advice_text,      
                color=self._color_by_score(100),
                title="버피",
            )

        # 유지
        self._bp_state = state
        self._bp_prev_s = s
        self.prev_label = "burpee"
        return None


    
    # ---------- pushup ----------
# 상단 import


   # 상단 import
    
    def _update_pushup(self, meta: Dict[str, Any]) -> Optional[EvalResult]:

        # return EvalResult(
        #     rep_inc=1,
        #     score=100,
        #     advice="너무 못한다ㅋ",
        #     color=self._color_by_score(100),   # ← base의 공통 함수 사용
        # )
        #오른쪽 값만 사용
        e = meta.get("elbow_r_deg")   # 필수
        k = meta.get("knee_r_deg")    # 선택
 
        # --- 안전화 ---
        try:
            e = float(e)
        except Exception:
            self.prev_label = "pushup"; return None
        if not math.isfinite(e):
            self.prev_label = "pushup"; return None
        if k is not None:
            try: k = float(k)
            except Exception: k = None

        # 기준선(히스테리시스)  ← 조정 포인트
        DOWN_ENTER    = 120.0   # ↓ 이하면 DOWN 진입
        SCORE_TRIGGER = 135.0   # ↑ 이상이면 UP 복귀 + 점수 확정 (기존 130.0 → 135.0 권장)

        # --- 상태머신 ---
        if self._pu_state == "UP":
            # 디바운스: 경계 근처 노이즈 방지
            if e <= DOWN_ENTER:
                self._deb = getattr(self, "_deb", 0) + 1
                if self._deb >= 3:                    # DEBOUNCE_N=3
                    self._pu_state = "DOWN"
                    self._pu_min_elbow = e
                    self._pu_min_knee  = k
                    self._deb = 0
            else:
                self._deb = 0

            self.prev_label = "pushup"
            return None

        # state == "DOWN": 한 rep 동안 최소값 갱신
        if e < self._pu_min_elbow:
            self._pu_min_elbow = e
        if k is not None:
            if (self._pu_min_knee is None) or (k < self._pu_min_knee):
                self._pu_min_knee = k

        # UP 복귀 → rep 인정 + 점수 산정
        if e >= SCORE_TRIGGER:
            self._deb = getattr(self, "_deb", 0) + 1
            if self._deb < 3:
                self.prev_label = "pushup"
                return None
            self._deb = 0

            em = self._pu_min_elbow
            km = self._pu_min_knee

            # 팔꿈치 점수: 64.3 → 100, 133.5 → 0 (작을수록 좋음)
            if em < 90.3:  em = 90.3
            if em > 133.5: em = 133.5
            elbow_s = (133.5 - em) / (133.5 - 90.3) * 100.0

            # 무릎 점수: 90 → 0, 157 → 100 (클수록 좋음)
            if km is not None:
                if km < 90:  km = 90#90.0
                if km > 157.0: km = 157.0
                knee_s = (km - 90.0) / (157.0 - 90.0) * 100.0
            else:
                knee_s = None

            # 가중합 (팔 70%, 무릎 30%)
            if knee_s is None:
                final_score = int(round(elbow_s))
            else:
                final_score = int(round(0.7 * elbow_s + 0.3 * knee_s))

            # 스코어 클램프
            final_score = 0 if final_score < 0 else (100 if final_score > 100 else final_score)

            # 피드백
            elbow_flare   = bool(meta.get("elbow_flare", False))
            torso_jitter  = float(meta.get("torso_jitter", 0.0))
            lumbar_ext    = float(meta.get("lumbar_ext", 0.0))

            # [ADDED] 간단 임계치(필요시 조정)
            TILT_LIMIT   = 5.0
            LUMBAR_LIMIT = 10.0

            # [ADDED] advice.py 에 넘길 컨텍스트
            ctx = {
                "elbow_not_deep":   (elbow_s < 60.0),
                "elbow_flare":      elbow_flare,
                "knee_more_extend": (knee_s is not None and knee_s < 80),
                "tilt_instability": (torso_jitter > TILT_LIMIT),
                "back_arch":        (lumbar_ext  > LUMBAR_LIMIT),
            }

            # [CHANGED] 문자열 직접 조합 → advice.py 사용(직전 문구 제외 + 랜덤 + 팁 결합)
            advice_text = get_advice("pushup", final_score, ctx)

            # 다음 rep 준비
            self._pu_state = "UP"
            self._pu_min_elbow = None
            self._pu_min_knee  = None

            return EvalResult(
                rep_inc=1,
                score=final_score,
                advice=advice_text,
                color=self._color_by_score(final_score),   # ← base의 공통 함수 사용
            )

        self.prev_label = "pushup"
        return None


    # ---------- jumping jack ----------
    
    def _update_Jumping_jack(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        # --- 각도 가져오기 ---
        s = meta.get("shoulder_avg_deg")
        if s is None:
            s = _avg_lr(meta, "shoulder")
        try:
            s = float(s)
        except Exception:
            self.prev_label = "jumping_jack"; return None
        if not math.isfinite(s):
            self.prev_label = "jumping_jack"; return None

        # --- 문턱(필요하면 숫자만 바꿔) ---
        OPEN_ENTER  = 90.0   # 이 이상이면 OPEN
        CLOSE_ENTER = 90.0   # 이 이하면 CLOSE

        # --- 상태 ---
        state   = getattr(self, "_jj_state", "CLOSE")   # "CLOSE" / "OPEN"
        cycles  = getattr(self, "_jj_cycles", 0)        # CLOSE 복귀 횟수
        prev_s  = getattr(self, "_jj_prev_s", s)        # 직전 각도

        if state == "CLOSE":
            # 위 문턱 '지나가는 순간'에만 OPEN (멈춤 필요 없음)
            if (prev_s < OPEN_ENTER) and (s >= OPEN_ENTER):
                state = "OPEN"
            self._jj_state = state
            self._jj_cycles = cycles
            self._jj_prev_s = s
            self.prev_label = "jumping_jack"
            return None

        # state == "OPEN"
        # 아래 문턱 '지나가는 순간'에만 CLOSE
        if (prev_s > CLOSE_ENTER) and (s <= CLOSE_ENTER):
            state = "CLOSE"
            cycles += 1
            self._jj_state = state
            self._jj_cycles = cycles
            self._jj_prev_s = s
            self.prev_label = "jumping_jack"

            # 두 번에 한 번만 rep +1
            if (cycles % 2) == 0:
                return EvalResult(
                    rep_inc=1,
                    score=100,
                    advice="굿.",
                    color=self._color_by_score(100),
                    title="점핑 잭",
                )
            return None

        # 전이 없음 → 값만 저장
        self._jj_state = state
        self._jj_cycles = cycles
        self._jj_prev_s = s
        self.prev_label = "jumping_jack"
        return None



def _color(s: int) -> QColor:
    if s is None: return QColor(200, 200, 200)
    if s >= 95:  return QColor(0, 200, 0)
    if s >= 80:  return QColor(255, 215, 0)
    return QColor(255, 80, 80)