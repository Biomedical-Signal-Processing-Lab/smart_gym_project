# app/core/evaluators/pose_angles.py
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import math

__all__ = [
    "compute_joint_angles",
    "update_meta_with_angles",
]

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC (at B) in degrees, range 0~180."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return np.nan
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = max(-1.0, min(1.0, cosv))  # numeric clamp
    return float(np.degrees(np.arccos(cosv)))

# app/core/evaluators/pose_angles.py
def _ok(kxy: np.ndarray, kcf: np.ndarray, idxs, thr: float) -> bool:
    # 좌표 음수여도 유효(크롭/오프셋 가능). NaN/Inf만 배제 + conf 체크
    return all(
        np.isfinite(kxy[i, 0]) and np.isfinite(kxy[i, 1]) and float(kcf[i]) >= thr
        for i in idxs
    )

def _idx_map(n):
    """COCO-17 vs BlazePose-33 자동 대응"""
    if n >= 33:  # BlazePose/Mediapipe 계열
        return dict(LSh=11, RSh=12, LEl=13, REl=14, LWr=15, RWr=16,
                    LHp=23, RHp=24, LKn=25, RKn=26, LAn=27, RAn=28)
    else:        # COCO-17 계열
        return dict(LSh=5,  RSh=6,  LEl=7,  REl=8,  LWr=9,  RWr=10,
                    LHp=11, RHp=12, LKn=13, RKn=14, LAn=15, RAn=16)

def compute_joint_angles(kxy: np.ndarray, kcf: np.ndarray, conf_thr: float = 0.2) -> Dict[str, float]:
    idx = _idx_map(kxy.shape[0])
    LSh,RSh,LEl,REl,LWr,RWr,LHp,RHp,LKn,RKn,LAn,RAn = (
        idx["LSh"],idx["RSh"],idx["LEl"],idx["REl"],idx["LWr"],idx["RWr"],
        idx["LHp"],idx["RHp"],idx["LKn"],idx["RKn"],idx["LAn"],idx["RAn"]
    )
    ang: Dict[str, float] = {}
    ang["Knee(L)"] = _angle_deg(kxy[LHp], kxy[LKn], kxy[LAn]) if _ok(kxy,kcf,[LHp,LKn,LAn],conf_thr) else np.nan
    ang["Knee(R)"] = _angle_deg(kxy[RHp], kxy[RKn], kxy[RAn]) if _ok(kxy,kcf,[RHp,RKn,RAn],conf_thr) else np.nan
    ang["Hip(L)"]  = _angle_deg(kxy[LSh], kxy[LHp], kxy[LKn]) if _ok(kxy,kcf,[LSh,LHp,LKn],conf_thr) else np.nan
    ang["Hip(R)"]  = _angle_deg(kxy[RSh], kxy[RHp], kxy[RKn]) if _ok(kxy,kcf,[RSh,RHp,RKn],conf_thr) else np.nan
    ang["Shoulder(L)"] = _angle_deg(kxy[LHp], kxy[LSh], kxy[LEl]) if _ok(kxy,kcf,[LHp,LSh,LEl],conf_thr) else np.nan
    ang["Shoulder(R)"] = _angle_deg(kxy[RHp], kxy[RSh], kxy[REl]) if _ok(kxy,kcf,[RHp,RSh,REl],conf_thr) else np.nan
    ang["Elbow(L)"] = _angle_deg(kxy[LSh], kxy[LEl], kxy[LWr]) if _ok(kxy,kcf,[LSh,LEl,LWr],conf_thr) else np.nan
    ang["Elbow(R)"] = _angle_deg(kxy[RSh], kxy[REl], kxy[RWr]) if _ok(kxy,kcf,[RSh,REl,RWr],conf_thr) else np.nan

    def _hip_line_angle(hip, knee):
        dx, dy = float(knee[0]-hip[0]), float(knee[1]-hip[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6: return np.nan
        return float(np.degrees(np.arctan2(abs(dy), abs(dx))))  # 0..90

    ang["HipLine(L)"] = _hip_line_angle(kxy[LHp], kxy[LKn]) if _ok(kxy,kcf,[LHp,LKn],conf_thr) else np.nan
    ang["HipLine(R)"] = _hip_line_angle(kxy[RHp], kxy[RKn]) if _ok(kxy,kcf,[RHp,RKn],conf_thr) else np.nan
    return ang


def update_meta_with_angles(
    meta: Dict,
    kxy: np.ndarray,
    kcf: np.ndarray,
    conf_thr: float = 0.2,
    ema: float = 0.0,
    prev: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    compute_joint_angles() 결과를 meta에 병합 + 좌우 평균 필드 추가.
    ema: 0~1, 지수이동평균(스무딩). 0이면 미사용.
    prev: 이전 프레임 각도 dict(ema>0일 때 사용)
    """
    ang = compute_joint_angles(kxy, kcf, conf_thr=conf_thr)

    # 좌우 평균(가능한 것만)
    def _avg_pair(a, b) -> float:
        av = ang.get(a, np.nan); bv = ang.get(b, np.nan)
        if not math.isnan(av) and not math.isnan(bv):
            return float((av + bv) / 2.0)
        if not math.isnan(av): return float(av)
        if not math.isnan(bv): return float(bv)
        return np.nan

    ang["Knee"]     = _avg_pair("Knee(L)", "Knee(R)")
    ang["Hip"]      = _avg_pair("Hip(L)", "Hip(R)")
    ang["Shoulder"] = _avg_pair("Shoulder(L)", "Shoulder(R)")
    ang["Elbow"]    = _avg_pair("Elbow(L)", "Elbow(R)")
    ang["HipLine"]  = _avg_pair("HipLine(L)", "HipLine(R)")

    # EMA 스무딩
    if ema > 0.0 and prev:
        out: Dict[str, float] = {}
        for k, v in ang.items():
            pv = prev.get(k, v) if prev else v
            if math.isnan(v):
                out[k] = pv  # 값이 없으면 이전값 유지
            else:
                out[k] = float(pv * (1.0 - ema) + v * ema)
        ang = out

    # NaN은 meta에 넣지 않음(평가기에서 None으로 인식)
    for k, v in list(ang.items()):
        if math.isnan(v):
            continue
        meta[k] = float(v)

    return ang
