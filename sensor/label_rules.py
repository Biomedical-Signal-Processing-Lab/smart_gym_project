import pandas as pd
import numpy as np
import math
import argparse

S_THRESH = 0.2       # |S| < 0.2 → Balanced
DT_SCL   = 120.0     # ms (tanh 스케일)
QH_SCL   = math.log(1.5)
EPS      = 1e-8

def tanh(x):
    return float(np.tanh(np.clip(x, -10, 10)))

def has_all_cols(df, cols):
    return all(c in df.columns for c in cols)

def safe_log_ratio(num, den):
    num = float(num) if pd.notna(num) else np.nan
    den = float(den) if pd.notna(den) else np.nan
    if not np.isfinite(num) or not np.isfinite(den) or num <= 0 or den <= 0:
        return np.nan
    return math.log(num/den)

def compute_S_full(row):
    """EMG 4채널 있을 때만 호출 (좌/우 평균)"""
    q_init = np.nanmean([row.get("quad_L_init_mean_desc"), row.get("quad_R_init_mean_desc")])
    h_init = np.nanmean([row.get("ham_L_init_mean_desc"),  row.get("ham_R_init_mean_desc")])
    q_on   = np.nanmean([row.get("quad_L_onset_desc"),     row.get("quad_R_onset_desc")])
    h_on   = np.nanmean([row.get("ham_L_onset_desc"),      row.get("ham_R_onset_desc")])

    if not np.isfinite(q_init) or not np.isfinite(h_init) or not np.isfinite(q_on) or not np.isfinite(h_on):
        return np.nan

    d_t  = q_on - h_on
    lnr  = safe_log_ratio(q_init, h_init)  # ln(Q/H)
    s_t  = tanh(d_t/DT_SCL)
    s_qh = tanh(lnr/QH_SCL) if np.isfinite(lnr) else np.nan
    return 0.6*s_t + (0.4*s_qh if np.isfinite(s_qh) else 0.0)

def compute_S_simple(row, z_env_init_mean):
    """
    1채널(ECG proxy)만 있을 때:
    - env_init_mean (초기 300ms) : 무릎(사두) 쪽 지표
    - env_int_desc vs env_int_asc 비율 : 하강/상승 전체 전략
    """
    env_init_z = z_env_init_mean  # 미리 계산한 z-score 전달
    # 하강/상승 적분 비율 -> ln(desc/asc)
    int_d = row.get("env_int_desc", np.nan)
    int_a = row.get("env_int_asc",  np.nan)
    lnr_int = np.nan
    if pd.notna(int_d) and pd.notna(int_a) and int_d > 0 and int_a > 0:
        lnr_int = math.log((float(int_d)+EPS)/(float(int_a)+EPS))

    s_init = 0.6 * tanh(env_init_z)                # 초기 강하면 Knee 쪽(+)
    s_int  = 0.4 * tanh(lnr_int/QH_SCL) if np.isfinite(lnr_int) else 0.0  # desc>asc면 Knee 쪽(+)로 가정
    return s_init + s_int

def label_from_S(s):
    if pd.isna(s):
        return "Unknown"
    if s >= +S_THRESH:
        return "Knee"
    if s <= -S_THRESH:
        return "Hip"
    return "Balanced"

def main(in_csv, out_csv):
    df = pd.read_csv(in_csv)

    # ----- 4채널 풀 규칙 가능 여부 확인
    full_cols = [
        "quad_L_init_mean_desc","quad_R_init_mean_desc",
        "ham_L_init_mean_desc","ham_R_init_mean_desc",
        "quad_L_onset_desc","quad_R_onset_desc",
        "ham_L_onset_desc","ham_R_onset_desc",
    ]
    can_full = has_all_cols(df, full_cols)

    S = []

    if can_full:
        # 정식 규칙
        for _, row in df.iterrows():
            S.append(compute_S_full(row))
    else:
        # 간이 규칙: env_init_mean z-score + 적분 비율
        if "env_init_mean" not in df.columns:
            raise ValueError("env_init_mean 컬럼이 없습니다. postprocess 단계에서 추가되었는지 확인하세요.")
        x = df["env_init_mean"].astype(float)
        mu, sd = float(x.mean()), float(x.std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        z_all = (x - mu)/sd

        for (_, row), z in zip(df.iterrows(), z_all):
            S.append(compute_S_simple(row, z))

    df["S_rule"] = S
    df["strategy_label"] = df["S_rule"].apply(label_from_S)
    df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  default="per_rep_features.csv")
    ap.add_argument("--out_csv", default="rep_features_labeled.csv")
    args = ap.parse_args()
    main(args.in_csv, args.out_csv)
