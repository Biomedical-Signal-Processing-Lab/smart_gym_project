# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import time
from collections import deque
import numpy as np
import cv2
from gi.repository import Gst

import settings as S
from pipeline import create_pipeline_and_sink
from display import make_window, put_info_bar
from meta import meta_counts_string, draw_kpt_indices

def _warn_paths():
    for pth, name in [(S.HEF, "HEF"), (S.POST_SO, "POST_SO"), (S.CROPPER_SO, "CROPPER_SO")]:
        if not os.path.isfile(pth):
            print(f"[WARN] {name} not found: {pth}")

def main():
    _warn_paths()
    pipe, sink = create_pipeline_and_sink(S)
    pipe.set_state(Gst.State.PLAYING)
    make_window(S.WINDOW_TITLE, S.FULLSCREEN, S.WINDOW_SIZE)

    # ===== 실제 표시 FPS 측정(1초 고정창) =====
    present_times = deque()     # time.monotonic() 타임스탬프(최근 1초)
    WINDOW_SEC = 1.0
    last_counted_pts = None     # 마지막으로 "카운트한" 프레임 PTS
    fps_disp = 0.0

    # 상태
    last_dbg_t = 0.0
    mirror = False
    show_kpt_idx = False
    debug = True
    last_raw = None

    try:
        while True:
            sample = sink.emit("try-pull-sample", 30_000_000)  # 30ms
            if sample is None:
                if last_raw is not None:
                    disp = cv2.flip(last_raw, 1) if mirror else last_raw
                    cv2.imshow(S.WINDOW_TITLE, disp)
                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key in (ord('a'), ord('A')): mirror = not mirror
                elif key in (ord('k'), ord('K')): show_kpt_idx = not show_kpt_idx
                elif key in (ord('d'), ord('D')): debug = not debug
                continue

            buf  = sample.get_buffer()
            caps = sample.get_caps()
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                continue

            try:
                w = caps.get_structure(0).get_value('width')
                h = caps.get_structure(0).get_value('height')
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                frame = arr.reshape((h, w, 3)).copy()  # overlay 결과(BGR)

                # (옵션) 키포인트 인덱스
                if show_kpt_idx:
                    draw_kpt_indices(frame, buf, w, h)

                # ---- 실제로 "새 프레임" 그렸는지 판단: PTS가 바뀌었을 때만 카운트 ----
                pts = buf.pts
                drew_new = (pts is not None and pts != Gst.CLOCK_TIME_NONE and pts != last_counted_pts)
                if drew_new:
                    last_counted_pts = pts
                    now = time.monotonic()
                    present_times.append(now)

                # 1초 창 유지(항상 분모 1.0초로 고정 → 과대표기 방지)
                now_chk = time.monotonic()
                cutoff = now_chk - WINDOW_SEC
                while present_times and present_times[0] < cutoff:
                    present_times.popleft()
                fps_disp = float(len(present_times)) / WINDOW_SEC

                # 소스 FPS 상한 클램프(드라이버 타임스탬프 경계 효과 방지)
                try:
                    if fps_disp > float(S.SRC_FPS):
                        fps_disp = float(S.SRC_FPS)
                except Exception:
                    pass

                # 상단 정보 바: 실제 표시 FPS만 표기
                put_info_bar(frame, f"FPS {fps_disp:.1f} | {w}x{h}", show=S.SHOW_INFO_BAR)

                # 1초 간격 콘솔 디버그
                if debug and (time.time() - last_dbg_t) >= 1.0:
                    print(meta_counts_string(buf, w, h, fps_disp))
                    last_dbg_t = time.time()

                last_raw = frame
                disp = cv2.flip(last_raw, 1) if mirror else last_raw
                cv2.imshow(S.WINDOW_TITLE, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key in (ord('a'), ord('A')): mirror = not mirror
                elif key in (ord('k'), ord('K')): show_kpt_idx = not show_kpt_idx
                elif key in (ord('d'), ord('D')): debug = not debug

            finally:
                buf.unmap(mapinfo)

    finally:
        pipe.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
