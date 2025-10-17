# app/core/evaluators/core_full.py
from __future__ import annotations
from typing import Dict, Any, Optional, Deque, Tuple
from collections import deque
import math

from .base import ExerciseEvaluator, EvalResult

class CoreFullEvaluator(ExerciseEvaluator):
    """
    버피(좌표 기반) 평가기
    - 숫자 새는 기준: 팔을 높이 들었을 때(손이 눈높이 위로 올라갔다가 다시 내려왔을 때 1회)
    - 점수 시점: 위 조건이 성립하는 순간에 '점프 높이'로 점수 산출
    - 추가 조건: 어깨 각도 ≥ 110° 이면서, 최근 프레임에서 '감소 → 증가' (turning point) 패턴
    """

    # ---- 튜닝 파라미터 (현장에 맞게 조정) ----
    EMA_BASELINE = 0.10          # 발목 바닥 기준선 EMA
    SHOULDER_MIN_DEG = 110.0     # 어깨 각도 최소
    EYE_FALLBACK_OFFSET = -30.0   # 눈 좌표가 없을 때 어깨 y + 오프셋(픽셀, y가 작을수록 위)
    DERIV_WIN = 5                # 어깨 각도 추세 판단용 윈도 길이(프레임)

    # 점프 높이 점수화 (신장 정규화 비율 = (발목_baseline_y - 발목_min_y)/body_scale)
    # body_scale: (힙-발목) 세로거리 평균
    JUMP_THRESHOLDS = {
        "best": 0.18,   # ≥ best  -> 100
        "max":  0.12,   # [max, best) -> 80
        "min":  0.05,   # [min, max)  -> 80
        #  그 외 -> 50
    }

    def __init__(self, label: str = "burpee"):
        super().__init__()
        self.mode = label
        self.reset()

    def reset(self) -> None:
        self.state = "READY"          # READY -> ARMS_UP -> READY
        self._deb = 0
        self._shoulder_hist: Deque[float] = deque(maxlen=self.DERIV_WIN)
        self._ankle_baseline_y: Optional[float] = None   # 바닥 기준선(크게 변동 X)
        self._min_ankle_y_during_up: Optional[float] = None

    # ------------- 좌표 헬퍼 -------------
    def _get_kpt(self, meta: Dict[str, Any]) -> Optional[list]:
        # ExercisePage에서 meta["_kpt"]에 kpt를 넣어주게 패치 필요(아래 2번 참고)
        return meta.get("_kpt")

    def _xy(self, kpt, idx) -> Optional[Tuple[float,float]]:
        if not kpt or idx >= len(kpt): return None
        p = kpt[idx]
        if p is None or len(p) < 2: return None
        return float(p[0]), float(p[1])

    # MediaPipe/BlazePose 인덱스(33 기준) - 프로젝트 인덱스 다르면 여기만 수정
    LM = dict(
        L_WRIST=15, R_WRIST=16,  # wrist
        L_EYE=2,   R_EYE=5,      # eye (없으면 nose/shoulder fallback)
        NOSE=0,
        L_SHOULDER=11, R_SHOULDER=12,
        L_HIP=23,  R_HIP=24,
        L_ANKLE=27, R_ANKLE=28,
        L_KNEE=25,  R_KNEE=26,
    )

    def _eye_level_y(self, kpt, meta) -> Optional[float]:
        # 눈 y (없으면 코 → 어깨 평균 + 오프셋)
        for key in ("L_EYE","R_EYE","NOSE"):
            idx = self.LM.get(key)
            if idx is not None:
                xy = self._xy(kpt, idx)
                if xy: return xy[1]
        # fallback: 어깨 평균 + 오프셋(픽셀)
        lsh = self._xy(kpt, self.LM["L_SHOULDER"])
        rsh = self._xy(kpt, self.LM["R_SHOULDER"])
        if lsh and rsh:
            return (lsh[1] + rsh[1]) / 2.0 + self.EYE_FALLBACK_OFFSET
        return None

    def _hands_y(self, kpt) -> Optional[float]:
        lw = self._xy(kpt, self.LM["L_WRIST"]); rw = self._xy(kpt, self.LM["R_WRIST"])
        if lw and rw:
            return min(lw[1], rw[1])  # y가 작을수록 화면상 더 위
        return lw[1] if lw else (rw[1] if rw else None)

    def _shoulder_avg_deg(self, meta) -> Optional[float]:
        # angles는 pose_angles.update_meta_with_angles에서 채워짐
        sL = meta.get("Shoulder(L)"); sR = meta.get("Shoulder(R)")
        if sL is None and sR is None: return None
        if sL is None: return float(sR)
        if sR is None: return float(sL)
        return float((sL + sR) / 2.0)

    def _ankle_avg_y(self, kpt) -> Optional[float]:
        la = self._xy(kpt, self.LM["L_ANKLE"]); ra = self._xy(kpt, self.LM["R_ANKLE"])
        if la and ra: return (la[1] + ra[1]) / 2.0
        return la[1] if la else (ra[1] if ra else None)

    def _body_scale(self, kpt) -> Optional[float]:
        # 힙-발목 세로거리 평균을 스케일로 사용 (픽셀)
        lh = self._xy(kpt, self.LM["L_HIP"]); la = self._xy(kpt, self.LM["L_ANKLE"])
        rh = self._xy(kpt, self.LM["R_HIP"]); ra = self._xy(kpt, self.LM["R_ANKLE"])
        vals = []
        if lh and la: vals.append(abs(la[1] - lh[1]))
        if rh and ra: vals.append(abs(ra[1] - rh[1]))
        if not vals: return None
        return float(sum(vals) / len(vals))

    def _shoulder_turning_up(self) -> bool:
        """
        최근 shoulder 히스토리에서 '감소 -> 증가' (U자 반전) 패턴인지 확인
        """
        h = list(self._shoulder_hist)
        if len(h) < max(3, self.DERIV_WIN // 2):
            return False
        # 간단한 로컬 최소 검출: h[k-1] > h[k] < h[k+1]
        k = len(h) - 2
        return h[k-1] > h[k] < h[k+1]

    # ------------- 점수화 -------------
    def _score_by_jump(self, jump_ratio: float) -> int:
        th = self.JUMP_THRESHOLDS
        if jump_ratio >= th["best"]: return 100
        if jump_ratio >= th["max"]:  return 80
        if jump_ratio >= th["min"]:  return 80
        return 50

    # ------------- 메인 업데이트 -------------
     # ✅ ExercisePage 표준 엔트리
    def update(self, meta: Dict[str, Any]) -> Optional[EvalResult]:
        if self.mode != "burpee":
            return None
        return self._update_burpee(meta)
    

    def _update_burpee(self, meta: Dict[str, Any], label) -> Optional[EvalResult]:
        if label:
            self.mode = label
        if self.mode != "burpee":
            return None

        kpt = self._get_kpt(meta)
        if not kpt: 
            return None

        # 좌표 기반 측정
        hand_y = self._hands_y(kpt)
        eye_y  = self._eye_level_y(kpt, meta)
        sh_deg = self._shoulder_avg_deg(meta)
        if hand_y is None or eye_y is None or sh_deg is None:
            return None

        # 어깨 각도 이력(추세 체크)
        self._shoulder_hist.append(sh_deg)

        # 발목 기준선(바닥) 업데이트 (READY 상태에서만 EMA)
        ankle_y = self._ankle_avg_y(kpt)
        if ankle_y is not None:
            if self._ankle_baseline_y is None:
                self._ankle_baseline_y = ankle_y
            elif self.state == "READY":
                a = self.EMA_BASELINE
                self._ankle_baseline_y = (1 - a) * self._ankle_baseline_y + a * ankle_y

        # ---- 상태머신 ----
        # y는 아래로 증가하므로, "손이 눈보다 위" == hand_y < eye_y
        hands_above_eye = (hand_y < eye_y)
        shoulder_ok = (sh_deg >= self.SHOULDER_MIN_DEG)

        if self.state == "READY":
            # 팔을 높이 든 상태에 진입
            if hands_above_eye and shoulder_ok:
                if self._shoulder_turning_up():     # 감소→증가 패턴
                    self.state = "ARMS_UP"
                    self._min_ankle_y_during_up = ankle_y  # 점프 최고점 탐지용 초기화
        elif self.state == "ARMS_UP":
            # 최고점 추적(작을수록 위쪽)
            if ankle_y is not None:
                if self._min_ankle_y_during_up is None:
                    self._min_ankle_y_during_up = ankle_y
                else:
                    self._min_ankle_y_during_up = min(self._min_ankle_y_during_up, ankle_y)

            # 손이 눈높이 아래로 떨어지면 1회 완료 + 점수
            if not hands_above_eye:
                self.state = "READY"
                # 점수 산출: 점프 높이
                score = None
                advice = "팔을 높이 들고 점프 높이를 늘려보세요."
                if self._ankle_baseline_y is not None and self._min_ankle_y_during_up is not None:
                    body_scale = self._body_scale(kpt)
                    if body_scale and body_scale > 1e-6:
                        # baseline_y가 더 큼(아래), min_y가 더 작음(위)
                        jump_pix = self._ankle_baseline_y - self._min_ankle_y_during_up
                        jump_ratio = max(0.0, jump_pix / body_scale)
                        score = self._score_by_jump(jump_ratio)
                        advice = f"추정 점프: {jump_ratio*100:.1f}% 신장"

                # 기본값(점프 추정 실패 시)
                if score is None:
                    score = 50

                # 1회 카운트 + 점수 반환
                return EvalResult(reps_delta=1, score=int(score), advice=advice)

        return None