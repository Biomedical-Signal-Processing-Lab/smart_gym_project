
import sys
import time
import csv
import serial

"""
Robust Arduino serial → CSV logger
- Handles old 4-field format:
  timestamp_ms,imu_pitch_deg,imu_pitch_vel,ecg_raw
- And new 5-field format (with raw + corrected pitch):
  timestamp_ms,imu_pitch_deg_raw,imu_pitch_deg,imu_pitch_vel,ecg_raw

Usage:
  python collect_serial.py COM3 115200 out.csv
  # or on Linux/macOS:
  python collect_serial.py /dev/ttyACM0 115200 out.csv
"""

def open_serial(port, baud):
    ser = serial.Serial(port, baud, timeout=1)
    # Give Arduino time to reset after opening the port
    time.sleep(2.0)
    # Flush any junk
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser

def main():
    if len(sys.argv) < 4:
        print("Usage: python collect_serial.py <PORT> <BAUD> <OUT_CSV>")
        sys.exit(1)

    port = sys.argv[1]
    baud = int(sys.argv[2])
    out_csv = sys.argv[3]

    ser = open_serial(port, baud)

    # Prepare CSV file & writer (streaming, no big RAM buffer)
    fout = open(out_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(fout)

    header_written = False
    n_rows = 0
    bad_lines = 0

    print("Listening... (Ctrl+C to stop)")

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Skip Arduino banner or comments starting with '#'
            if line.startswith("#"):
                continue

            # Write header transparently if Arduino sends it
            if not header_written and "timestamp" in line:
                cols = [c.strip() for c in line.split(",")]
                writer.writerow(cols)
                header_written = True
                fout.flush()
                print("Header:", cols)
                continue

            # Parse data rows (support 4 or 5 columns)
            parts = [p.strip() for p in line.split(",")]
            try:
                if len(parts) == 4:
                    t, pitch, pitch_vel, ecg = parts
                    row = [int(t), float(pitch), float(pitch_vel), int(ecg)]
                    # If 4-field header wasn't seen, write one now
                    if not header_written:
                        writer.writerow(["timestamp_ms","imu_pitch_deg","imu_pitch_vel","ecg_raw"])
                        header_written = True
                    writer.writerow(row)

                elif len(parts) == 5:
                    t, pitch_raw, pitch_corr, pitch_vel, ecg = parts
                    row = [int(t), float(pitch_raw), float(pitch_corr), float(pitch_vel), int(ecg)]
                    if not header_written:
                        writer.writerow(["timestamp_ms","imu_pitch_deg_raw","imu_pitch_deg","imu_pitch_vel","ecg_raw"])
                        header_written = True
                    writer.writerow(row)

                else:
                    bad_lines += 1
                    if bad_lines <= 5:
                        print(f"[warn] malformed line ({len(parts)} fields):", line)
                    continue

                n_rows += 1
                if n_rows % 100 == 0:
                    fout.flush()

            except Exception as e:
                bad_lines += 1
                if bad_lines <= 5:
                    print(f"[warn] parse error: {e} | line: {line}")
                continue

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        ser.close()
        fout.flush()
        fout.close()
        print(f"Wrote {n_rows} rows to {out_csv}. Malformed lines: {bad_lines}")

if __name__ == "__main__":
    main()
