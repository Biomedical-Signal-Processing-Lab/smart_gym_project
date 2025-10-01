import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt

# ===== 설정 =====
INIT_MS = 300                # 초기 윈도우(ms)
USE_DESC_START = False       # True로 하면 하강 시작(state==DESC_VAL) 시점 기준
DESC_VAL = -1                # 하강 state 값: 너 환경에 맞게 -1 또는 +1

def bandpass_lowrect_env(x, fs, lo=20, hi=100, lpf=8):
    x = x.astype(float)
    x -= np.median(x)
    b,a = butter(4, [lo/(0.5*fs), hi/(0.5*fs)], btype="band")
    y = filtfilt(b,a,x) if len(x) > 20 else x
    y = np.abs(y)
    bL,aL = butter(2, lpf/(0.5*fs), btype="low")
    env = filtfilt(bL,aL,y) if len(y) > 10 else y
    return env

def segment_by_rep_id(df):
    reps = []
    rep_ids = df["rep_id"].values
    if rep_ids.max() <= 0:
        # fallback: DESC(-1)→ASC(+1)→STOP(0) 패턴으로 스캔
        state = df["state"].values
        i = 1; n = len(state)
        while i < n:
            while i < n and state[i] != DESC_VAL: i += 1
            if i >= n: break
            start = i
            sawAsc=False
            while i < n and state[i] == DESC_VAL: i += 1   # 하강 구간 지나감
            while i < n and state[i] != DESC_VAL:
                if state[i] == (-DESC_VAL): sawAsc=True    # 상승 감지
                i += 1
                if sawAsc and (i==n or state[i]==DESC_VAL):
                    reps.append((start, i-1)); break
        return reps
    # rep_id가 증가할 때마다 닫기
    cur_id = rep_ids[0]
    start = 0
    for i in range(1, len(df)):
        if rep_ids[i] != cur_id:
            reps.append((start, i-1))
            start = i
            cur_id = rep_ids[i]
    if start < len(df):
        reps.append((start, len(df)-1))
    return reps

def main(csv_in, fs_ecg=500, outdir="."):
    df = pd.read_csv(csv_in)

    # ECG envelope
    if "ecg_env" not in df.columns:
        df["ecg_env"] = bandpass_lowrect_env(df["ecg_raw"].to_numpy(), fs_ecg)

    # rep segments
    reps = segment_by_rep_id(df)

    # arrays
    ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
    env = df["ecg_env"].to_numpy(dtype=float)
    pitch = df["imu_pitch_deg"].to_numpy(dtype=float)
    state = df["state"].to_numpy(dtype=int)

    feats = []
    for (i0, i1) in reps:
        t0, t1 = ts[i0], ts[i1]
        seg_state = state[i0:i1+1]
        seg_pitch = pitch[i0:i1+1]
        seg_env = env[i0:i1+1]
        seg_ts = ts[i0:i1+1]

        # tempos (ms 합)
        dt_ms = np.diff(seg_ts, prepend=seg_ts[0])
        tempo_desc = float(np.sum(dt_ms[seg_state == DESC_VAL])) if np.any(seg_state == DESC_VAL) else 0.0
        tempo_asc  = float(np.sum(dt_ms[seg_state == -DESC_VAL])) if np.any(seg_state == -DESC_VAL) else 0.0

        # ROM
        rom = float(np.max(seg_pitch) - np.min(seg_pitch)) if seg_pitch.size else 0.0

        # integrals / peaks (ECG env)
        def seg_int(mask, fs=100):
            if not np.any(mask): return 0.0
            idx = np.where(mask)[0]
            return float(np.trapz(seg_env[idx[0]:idx[-1]+1], dx=1.0/fs))

        int_desc = seg_int(seg_state == DESC_VAL)
        int_asc  = seg_int(seg_state == -DESC_VAL)
        pk_desc = float(np.max(seg_env[seg_state == DESC_VAL])) if np.any(seg_state == DESC_VAL) else float(np.max(seg_env)) if seg_env.size else 0.0
        pk_asc  = float(np.max(seg_env[seg_state == -DESC_VAL])) if np.any(seg_state == -DESC_VAL) else float(np.max(seg_env)) if seg_env.size else 0.0

        # 초기 300ms 평균값 (기본: rep 시작 기준)
        if USE_DESC_START:
            idx_desc = np.where(seg_state == DESC_VAL)[0]
            if idx_desc.size > 0:
                t_start = int(seg_ts[idx_desc[0]])             # 하강 시작 시각
            else:
                t_start = int(t0)                               # fallback
        else:
            t_start = int(t0)                                   # rep 시작 시각

        win_end = min(t_start + INIT_MS, int(t1))
        mask_init = (seg_ts >= t_start) & (seg_ts <= win_end)
        env_init_mean = float(np.mean(seg_env[mask_init])) if np.any(mask_init) else np.nan

        feats.append({
            "rep_start_ms": float(t0),
            "rep_end_ms": float(t1),
            "tempo_desc_ms": tempo_desc,
            "tempo_asc_ms": tempo_asc,
            "rom_deg": rom,
            "env_int_desc": float(int_desc),
            "env_int_asc": float(int_asc),
            "env_peak_desc": pk_desc,
            "env_peak_asc": pk_asc,
            "env_init_mean": env_init_mean,   # ★ 추가된 컬럼
        })

    # save
    out1 = f"{outdir.rstrip('/')}/per_sample_processed.csv"
    out2 = f"{outdir.rstrip('/')}/per_rep_features.csv"
    df.to_csv(out1, index=False)
    pd.DataFrame(feats).to_csv(out2, index=False)
    print("saved:", out1, out2)

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fs_ecg", type=int, default=500)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    main(args.csv, fs_ecg=args.fs_ecg, outdir=args.outdir)
