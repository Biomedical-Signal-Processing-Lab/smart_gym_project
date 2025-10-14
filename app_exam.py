# app_exam.py
# -*- coding: utf-8 -*-
"""
Example application:
- uses library to get (frame, people, cls_result)
- draws overlay in the app (not in the library)
- press 'q' to quit
"""
import cv2
import numpy as np
from lib.app_pose_stream import start_stream, read_latest, stop_stream
from lib import settings as S

# model paths (adjust as needed)
# ONNX = getattr(__import__("settings") , "TCN_ONNX", "models/251013_10_44_40/tcn.onnx")
# JSON = getattr(__import__("settings"), "TCN_JSON", "models/251013_10_44_40/tcn.json")

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Qt를 Wayland 대신 X11로 사용
os.environ["QT_NO_GLIB"] = "1"         # Qt↔GLib 통합 비활성화(충돌 회피)

ONNX = getattr(S , "TCN_ONNX", "models/251013_10_44_40/tcn.onnx")
JSON = getattr(S, "TCN_JSON", "models/251013_10_44_40/tcn.json")


# drawing settings
CONF_THR = 0.65
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
         (11,13),(13,15),(12,14),(14,16)]  # nose-shoulder edges removed

def draw_app_overlay(frame, people, cls):
    H, W = frame.shape[:2]
    max_len2 = (max(W, H) * 0.6) ** 2

    # draw all people with high-conf points
    for p in people:
        x1,y1,x2,y2 = map(int, p.get("bbox", [0,0,0,0]))
        pts = p.get("kpt", [])
        hi = [pt for pt in pts if len(pt)>=3 and float(pt[2])>=CONF_THR]
        color_box = (0,255,0) if len(hi)>=6 else (160,160,160)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color_box, 2)

        vis = [False]*len(pts)
        for j, pt in enumerate(pts):
            if len(pt) < 2: continue
            x, y = int(pt[0]), int(pt[1])
            c = float(pt[2]) if len(pt) >= 3 else 1.0
            if c < CONF_THR: continue
            vis[j] = True
            cv2.circle(frame, (x,y), 3, (255,255,255), -1)

        for a,b in EDGES:
            if a < len(pts) and b < len(pts) and vis[a] and vis[b]:
                x1_, y1_ = int(pts[a][0]), int(pts[a][1])
                x2_, y2_ = int(pts[b][0]), int(pts[b][1])
                dx, dy = x1_ - x2_, y1_ - y2_
                if (dx*dx + dy*dy) <= max_len2:
                    cv2.line(frame, (x1_, y1_), (x2_, y2_), (0,200,255), 2)

    # status text
    if cls is None:
        txt = "[TCN] no person"
    else:
        txt = f"[TCN] {cls['label']} ({cls['score']:.2f}) | conf_mean={cls.get('conf_mean', 0.0):.2f}"
    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)

def main():
    start_stream(ONNX, JSON, conf_thr=CONF_THR, stride=1)  # stream starts
    try:
        while True:
            frame, people, label = read_latest(timeout=0.2)
            if frame is None:
                continue
            # the app is in charge of drawing:
            draw_app_overlay(frame, people, label)

            cv2.imshow("preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stop_stream()
        try: cv2.destroyAllWindows()
        except Exception: pass

if __name__ == "__main__":
    main()