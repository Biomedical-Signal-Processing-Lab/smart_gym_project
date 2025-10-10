# ================================================================
# pi_ble_infer_server.py
# BLE로 수신한 NANO33_UART 6필드 데이터를 실시간 특징계산 → 모델추론
# - 입력 포맷(6필드): ts, pitch, vel, st, rep_id, ecg_raw
# - 적응형 envelope 추출 (fs에 안전하게 자동 조정)
# - rep 종료 강화: (1) rep_id 변화, (2) 움직임 후 정지 지속
# ================================================================

import argparse, asyncio, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from bleak import BleakClient, BleakScanner
from scipy.signal import butter, filtfilt
from datetime import datetime

# ===== 설정 =====
DEFAULT_NAME   = "NANO33_UART"
SERVICE_UUID   = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
TX_UUID        = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
FS_HINT        = 100.0     # 샘플링 추정 실패 시 fallback
INIT_MS        = 300       # 초기 300ms 평균
V_THRESH       = 10.0      # 속도 임계(하강/상승 진입)
V_HYST         = 5.0       # 히스테리시스
IDLE_END_MS    = 150.0     # 정지(0) 유지시 rep 종료로 판단하는 최소 시간

# ===== 모델/피처 =====
ART = Path("artifacts")
MODEL = joblib.load(ART / "best_model.joblib")
FEATURES = json.loads((ART / "feature_order.json").read_text(encoding="utf-8"))
CLASSES = None
try:
    CLASSES = MODEL.named_steps["clf"].classes_.tolist()
except Exception:
    pass

# ===== 로그 파일 =====
LOG_PATH = Path("per_rep_predictions.csv")
if not LOG_PATH.exists():
    LOG_PATH.write_text(
        "timestamp,rep_index,pred,"
        + ",".join([f"proba_{c}" for c in (CLASSES or [])]) + ","
        + ",".join(FEATURES) + "\n",
        encoding="utf-8"
    )

# ===== 신호 처리: 적응형 envelope =====
def bandpass_lowrect_env(x, fs, lo=20, hi=100, lpf=8):
    """
    ECG/EMG proxy -> envelope 추출 (fs에 맞춰 안전하게 자동 조정)
    - fs >= 220Hz: 20-100Hz bandpass → |·| → LPF(lpf Hz)
    - 120Hz <= fs < 220Hz: 10-45Hz
    - 60Hz  <= fs < 120Hz: 5-20Hz
    - fs < 60Hz: bandpass 생략(베이스라인 제거 후 |·| → LPF)
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 10:
        return np.maximum(0.0, x - np.median(x))

    x = x - np.median(x)  # baseline 제거

    # fs에 맞춰 안전한 통과대역 선택
    if fs >= 220:
        blo, bhi = 20.0, 100.0
    elif fs >= 120:
        blo, bhi = 10.0, 45.0
    elif fs >= 60:
        blo, bhi = 5.0, 20.0
    else:
        blo, bhi = None, None  # bandpass 생략

    y = x
    try:
        if blo is not None and bhi is not None:
            nyq = 0.5 * fs
            wn1 = max(1e-6, blo / nyq)
            wn2 = min(0.99,  bhi / nyq)
            if wn2 > wn1 and 0 < wn1 < 1 and 0 < wn2 < 1:
                b, a = butter(4, [wn1, wn2], btype="band")
                y = filtfilt(b, a, y) if len(y) > 20 else y
    except Exception:
        y = x  # 필터 실패 시 원신호로 진행

    # 정류
    y = np.abs(y)

    # 저역 통과로 envelope
    try:
        nyq = 0.5 * fs
        lpf_hz = min(lpf, 0.45 * fs)  # 과한 차단 방지
        wn = max(1e-6, min(0.99, lpf_hz / nyq))
        bL, aL = butter(2, wn, btype="low")
        env = filtfilt(bL, aL, y) if len(y) > 10 else y
    except Exception:
        env = y
    return env

def update_state(prev_state, v):
    # 속도 기반 보조 상태기계 (-1: 하강, +1: 상승, 0: 정지)
    if prev_state == 0:
        if v <= -V_THRESH: return -1
        if v >= +V_THRESH: return +1
        return 0
    if prev_state == -1:
        if -(V_THRESH - V_HYST) <= v <= (V_THRESH - V_HYST): return 0
        return -1
    if prev_state == +1:
        if -(V_THRESH - V_HYST) <= v <= (V_THRESH - V_HYST): return 0
        return +1
    return 0

def trapz_int(y, t_ms):
    if len(y) < 2: return 0.0
    t = np.asarray(t_ms, float) * 1e-3
    return float(np.trapz(y, t))

def compute_rep_features(buf):
    ts   = np.array([r["ts"] for r in buf], dtype=float)
    pitch= np.array([r["pitch"] for r in buf], dtype=float)
    vel  = np.array([r["vel"] for r in buf], dtype=float)
    ecg  = np.array([r["ecg"] for r in buf], dtype=float)
    state= np.array([r["state"] for r in buf], dtype=int)

    dt = np.diff(ts) * 1e-3
    fs = 1.0/np.median(dt) if len(dt)>0 and np.all(dt>0) else FS_HINT
    env = bandpass_lowrect_env(ecg, fs)

    desc_idx = np.where(state == -1)[0]
    asc_idx  = np.where(state == +1)[0]

    tempo_desc_ms = float(ts[desc_idx[-1]] - ts[desc_idx[0]]) if len(desc_idx) > 1 else 0.0
    tempo_asc_ms  = float(ts[asc_idx[-1]]  - ts[asc_idx[0]])  if len(asc_idx)  > 1 else 0.0
    rom_deg = float(np.nanmax(pitch) - np.nanmin(pitch))

    env_int_desc = trapz_int(env[desc_idx], ts[desc_idx]) if len(desc_idx)>1 else 0.0
    env_int_asc  = trapz_int(env[asc_idx],  ts[asc_idx])  if len(asc_idx)>1  else 0.0
    env_peak_desc= float(np.nanmax(env[desc_idx])) if len(desc_idx) else 0.0
    env_peak_asc = float(np.nanmax(env[asc_idx]))  if len(asc_idx)  else 0.0

    if len(desc_idx):
        t0 = ts[desc_idx[0]]
        init_mask = np.where((ts >= t0) & (ts <= t0 + INIT_MS))[0]
        env_init_mean = float(np.nanmean(env[init_mask])) if len(init_mask) else float(env[desc_idx[0]])
    else:
        env_init_mean = 0.0

    return {
        "tempo_desc_ms": tempo_desc_ms,
        "tempo_asc_ms":  tempo_asc_ms,
        "rom_deg":       rom_deg,
        "env_int_desc":  env_int_desc,
        "env_int_asc":   env_int_asc,
        "env_peak_desc": env_peak_desc,
        "env_peak_asc":  env_peak_asc,
        "env_init_mean": env_init_mean,
    }

def predict_features(feat: dict):
    df = pd.DataFrame([feat], columns=FEATURES)
    y_hat = MODEL.predict(df)[0]
    probs = {}
    try:
        proba = MODEL.predict_proba(df)[0]
        for c,p in zip(CLASSES, proba):
            probs[c] = float(p)
    except Exception:
        pass
    return y_hat, probs

# ============== REP TRACKER =================
class RepTracker:
    """rep_id 변화 + 움직임후 정지로 rep 종료 감지"""
    def __init__(self):
        self.state = 0
        self.last_state = 0
        self.buf = []
        self.current_rep_id = None
        self.rep_idx = 0
        self.idle_start_ts = None
        self.had_motion = False

    def _finalize(self):
        if len(self.buf) < 3:
            self.buf.clear()
            self.had_motion = False
            self.idle_start_ts = None
            return None
        self.rep_idx = (self.current_rep_id or (self.rep_idx + 1))
        feats = compute_rep_features(self.buf)
        self.buf.clear()
        self.had_motion = False
        self.idle_start_ts = None
        return feats

    def feed(self, ts_ms, pitch, vel, ecg, maybe_state=None, maybe_rep_id=None):
        # 상태: 아두이노 state 우선, 0이면 속도로 보조판정
        st = int(maybe_state) if maybe_state is not None else update_state(self.state, vel)
        if st == 0:
            st = update_state(self.state, vel)

        feats_to_return = None

        # rep_id 경계 우선
        if maybe_rep_id is not None:
            rpid = int(maybe_rep_id)
            if self.current_rep_id is None:
                self.current_rep_id = rpid
            elif rpid != self.current_rep_id:
                feats_to_return = self._finalize()
                self.current_rep_id = rpid

        # 샘플 추가
        self.state = st
        self.buf.append({"ts":ts_ms, "pitch":pitch, "vel":vel, "ecg":ecg, "state":st})

        # 움직임 있었는지 체크
        if st in (-1, +1):
            self.had_motion = True
            self.idle_start_ts = None

        # 움직임 후 정지 지속이면 종료
        if st == 0:
            if self.idle_start_ts is None:
                self.idle_start_ts = ts_ms
            idle_dur = ts_ms - self.idle_start_ts
            if self.had_motion and idle_dur >= IDLE_END_MS:
                feats_to_return = self._finalize()

        self.last_state = st
        return feats_to_return

# ============== BLE 루프 =================
async def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--addr", default="")
    args = parser.parse_args()

    want_name = (args.name or "").strip().lower()
    want_addr = (args.addr or "").strip().upper()

    print("[SCAN] BLE 기기 검색(8초)…")
    devices = await BleakScanner.discover(timeout=8.0)
    dev = None
    if want_addr:
        for d in devices:
            if (d.address or "").upper() == want_addr:
                dev = d; break
    else:
        for d in devices:
            n = (d.name or "").strip().lower()
            if want_name and want_name in n:
                dev = d; break
    if not dev:
        print("[SCAN] 발견 장치:")
        for d in devices:
            print(f"- {d.address}  name='{d.name}'")
        raise RuntimeError("대상 장치를 찾지 못했습니다. --addr 또는 --name 확인")

    tracker = RepTracker()
    line_buf = bytearray()

    def on_notify(_, data: bytearray):
        nonlocal line_buf
        line_buf.extend(data)
        while b"\n" in line_buf:
            raw, _, line_buf = line_buf.partition(b"\n")
            try:
                s = raw.decode("utf-8").strip()
                parts = s.split(",")
                feats = None
                if len(parts) == 6:
                    # ✅ 포맷: ts, pitch, vel, st, rep_id, ecg_raw
                    ts, pitch, vel, st, rep_id, ecg = parts
                    feats = tracker.feed(
                        float(ts), float(pitch), float(vel), float(ecg),
                        maybe_state=int(float(st)),
                        maybe_rep_id=int(float(rep_id))
                    )
                elif len(parts) == 4:
                    ts, pitch, vel, ecg = parts
                    feats = tracker.feed(float(ts), float(pitch), float(vel), float(ecg))
                else:
                    return

                if feats:
                    y, probs = predict_features(feats)
                    now = datetime.now().isoformat(timespec="seconds")
                    if probs:
                        order = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        top = f"{order[0][0]} {order[0][1]*100:.1f}%"
                        print(f"[{now}] Rep {tracker.rep_idx} → Pred={y} ({top})  {dict(order)}")
                    else:
                        print(f"[{now}] Rep {tracker.rep_idx} → Pred={y}")

                    row = [
                        now, tracker.rep_idx, y,
                        *([probs.get(c, np.nan) for c in (CLASSES or [])]),
                        *(feats[k] for k in FEATURES)
                    ]
                    with LOG_PATH.open("a", encoding="utf-8") as f:
                        f.write(",".join(str(x) for x in row) + "\n")

            except Exception as e:
                print(f"[WARN] parse fail: {e}")

    async with BleakClient(dev) as client:
        print("[BLE] 연결됨:", dev)
        await client.start_notify(TX_UUID, on_notify)
        print("[BLE] 수신 시작 (Ctrl+C 종료)")
        try:
            while True:
                await asyncio.sleep(0.25)
        except KeyboardInterrupt:
            print("\n[BLE] 종료 요청")
        finally:
            await client.stop_notify(TX_UUID)

if __name__ == "__main__":
    asyncio.run(run())

