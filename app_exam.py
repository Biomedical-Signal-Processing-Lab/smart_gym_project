# app_exam.py
# -*- coding: utf-8 -*-
"""
Example app (PC): YOLOv8m-pose ONNX + TCN classifier
- Library returns (frame, people, label); app draws overlay & shows preview.
"""

import os
# Stabilize Qt/GLib on some desktops (optional but helps)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_NO_GLIB", "1")

import cv2
import numpy as np
from lib.pc_pose_stream import start_stream, read_latest, stop_stream
from lib import settings as S
# Model paths (TCN)
_settings = S
ONNX = getattr(_settings, "TCN_ONNX", "models/251013_10_44_40/tcn.onnx")
JSON = getattr(_settings, "TCN_JSON", "models/251013_10_44_40/tcn.json")
YOLO_POSE_ONNX = getattr(_settings, "YOLO_POSE_ONNX", "models/yolov8m_pose.onnx")  # 있으면 사용

CONF_THR = 0.65

# Skeleton edges (nose-shoulder removed)
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
         (11,13),(13,15),(12,14),(14,16)]

def _clamp_xy(x, y, w, h):
    if x < 0: x = 0
    if y < 0: y = 0
    if x >= w: x = w - 1
    if y >= h: y = h - 1
    return int(x), int(y)

def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w-1))
    y2 = max(0, min(int(y2), h-1))
    # 만약 뒤집힌 좌표가 오면 바로잡기
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def draw_app_overlay(frame, people, label):
    H, W = frame.shape[:2]
    max_len2 = (max(W, H) * 0.6) ** 2
    conf_thr = CONF_THR

    for p in people:
        x1,y1,x2,y2 = _clamp_box(*p.get("bbox", [0,0,0,0]), W, H)
        pts = p.get("kpt", [])

        # bbox 색: 고신뢰 점이 충분히 있으면 초록
        hi = [pt for pt in pts if len(pt)>=3 and float(pt[2])>=conf_thr]
        color_box = (0,255,0) if len(hi)>=6 else (160,160,160)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color_box, 2)

        # 점
        vis = [False]*len(pts)
        for j, pt in enumerate(pts):
            if len(pt) < 2: continue
            x, y = _clamp_xy(pt[0], pt[1], W, H)
            c = float(pt[2]) if len(pt) >= 3 else 1.0
            if c < conf_thr: continue
            vis[j] = True
            cv2.circle(frame, (x,y), 3, (255,255,255), -1)

        # 엣지 (코-어깨 제거는 EDGES 정의로 처리)
        for a,b in EDGES:
            if a < len(pts) and b < len(pts) and vis[a] and vis[b]:
                x1_, y1_ = _clamp_xy(pts[a][0], pts[a][1], W, H)
                x2_, y2_ = _clamp_xy(pts[b][0], pts[b][1], W, H)
                dx, dy = x1_ - x2_, y1_ - y2_
                if (dx*dx + dy*dy) <= max_len2:
                    cv2.line(frame, (x1_, y1_), (x2_, y2_), (0,200,255), 2)

    txt = "[TCN] no person" if (label is None) else f"[TCN] {label}"
    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)

def main():
    start_stream(
        ONNX, JSON,
        conf_thr=CONF_THR, stride=1,
        yolo_onnx=YOLO_POSE_ONNX,  # 설정에 있으면 사용
        use_gpu=True,              # CUDA 있다면 사용(없으면 자동 CPU)
        gpu_id=0
    )

    try:
        while True:
            frame, people, label = read_latest(timeout=0.2)
            if frame is None:
                continue
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