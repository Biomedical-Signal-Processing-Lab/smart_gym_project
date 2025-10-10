import sys, csv, time, argparse
import serial

HEADER6 = ["timestamp_ms","pitch_deg","pitch_vel_dps","state","rep_id","ecg_raw"]

def parse_line_to_row(line: str):
    """CSV 한 줄을 파싱해서 6개 값으로 반환. 실패하면 None."""
    line = line.strip()
    if not line:
        return None
    # 헤더가 들어오면 무시
    if "timestamp_ms" in line and "pitch_deg" in line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None
    try:
        t = int(float(parts[0]))                 # timestamp_ms
        pitch = float(parts[1])                  # pitch_deg
        vel = float(parts[2])                    # pitch_vel_dps
        state = int(float(parts[3]))             # state
        rep = int(float(parts[4]))               # rep_id
        ecg = int(float(parts[5]))               # ecg_raw (ADC)
        return [t, pitch, vel, state, rep, ecg]
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Collect Arduino serial to CSV")
    ap.add_argument("port", help="Serial port (e.g., COM5 or /dev/ttyACM0)")
    ap.add_argument("baud", type=int, help="Baud rate (e.g., 115200)")
    ap.add_argument("out", help="Output CSV filename")
    ap.add_argument("--timeout", type=float, default=1.0, help="Serial timeout (s)")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    print("Listening... (Ctrl+C to stop)")

    rows = []
    malformed = 0
    try:
        while True:
            line = ser.readline().decode(errors="ignore")
            if not line:
                continue
            row = parse_line_to_row(line)
            if row is None:
                malformed += 1
                # 필요하면 주석 해제해 경고 표시
                # print(f"[warn] malformed: {line.strip()}")
                continue
            rows.append(row)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    # CSV 저장
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER6)
        w.writerows(rows)

    print(f"\nStopped by user.")
    print(f"Wrote {len(rows)} rows to {args.out}. Malformed lines: {malformed}")

if __name__ == "__main__":
    main()
