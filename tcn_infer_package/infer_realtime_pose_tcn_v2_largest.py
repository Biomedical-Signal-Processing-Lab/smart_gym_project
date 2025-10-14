#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_realtime_pose_tcn_v2_largest.py
- 변경점: 다수 인원이 검출되면 **가장 큰 사람(바운딩박스 면적 최대)** 1명만 선택
- 추가 옵션:
    --person-mode largest        # (기본값) bbox 면적이 가장 큰 사람 1명만
    --min-area 15000             # 픽셀^2 기준 최소 면적 필터(작은 사람/배경 오검출 제거)

    
python infer_realtime_pose_tcn_v2_largest.py \
  --ckpt ./best.pt \
  --model ./yolov8m_pose.onnx \
  --device 0 --imgsz 640 --conf 0.25 --draw \
  --preview-scale 0.6 \
  --min-area 20000



기타 동작은 infer_realtime_pose_tcn_v2.py 와 동일.
"""

import argparse, time, os, collections
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import torch.nn as nn

# ----- TCN (same as training) -----
class Chomp1d(nn.Module):
    def __init__(self, chomp_size): super().__init__(); self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding); self.relu1 = nn.ReLU(); self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding); self.relu2 = nn.ReLU(); self.drop2 = nn.Dropout(dropout)
        self.down = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x); out = self.chomp1(out); out = self.relu1(out); out = self.drop1(out)
        out = self.conv2(out); out = self.chomp2(out); out = self.relu2(out); out = self.drop2(out)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_channels, num_classes, num_levels=4, n_channels=128, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []; in_ch = input_channels
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, n_channels, kernel_size, 1, dilation, (kernel_size-1)*dilation, dropout))
            in_ch = n_channels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(n_channels, num_classes))
    def forward(self, x):
        y = self.backbone(x); return self.head(y)

# ----- Pose utils -----
KPTS = 17
COCO_SKELETON = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]

def normalize_xy(kxy, width, height, mode="image"):
    kxy = kxy.copy()
    if mode == "image":
        kxy[:,0] /= max(1, width); kxy[:,1] /= max(1, height); return kxy
    elif mode == "center_hips":
        cx = (kxy[11,0] + kxy[12,0]) / 2.0; cy = (kxy[11,1] + kxy[12,1]) / 2.0
        kxy[:,0] = (kxy[:,0] - cx) / max(1, width); kxy[:,1] = (kxy[:,1] - cy) / max(1, height); return kxy
    else:
        raise ValueError("unknown norm")

def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]; C = ckpt["input_channels"]; hp = ckpt.get("hparams", {})
    model = TCN(input_channels=C, num_classes=len(classes),
                num_levels=hp.get("levels",4), n_channels=hp.get("channels",128),
                kernel_size=hp.get("kernel",3), dropout=hp.get("dropout",0.2))
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, classes, C, hp

def choose_person_largest(res, min_area: float = 0.0):
    """Return index of the largest person by bbox area, or None if none passes min_area."""
    if (res.boxes is None) or (len(res.boxes) == 0):
        return None
    xywh = res.boxes.xywh.cpu().numpy()   # (N,4): x,y,w,h
    areas = xywh[:,2] * xywh[:,3]         # (N,)
    valid = np.where(areas >= float(min_area))[0] if min_area and min_area > 0 else np.arange(len(areas))
    if valid.size == 0:
        return None
    idx_local = int(np.argmax(areas[valid]))
    return int(valid[idx_local])

def draw_pose(img, kxy, draw=True):
    if not draw: return img
    for a,b in COCO_SKELETON:
        if a<0 or b<0 or a>=KPTS or b>=KPTS: continue
        pa = tuple(map(int, kxy[a])); pb = tuple(map(int, kxy[b]))
        cv2.line(img, pa, pb, (0,255,255), 2)
    for (x,y) in kxy:
        cv2.circle(img, (int(x),int(y)), 3, (255,0,255), -1)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=0, type=int)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--iou", default=0.5, type=float)
    ap.add_argument("--video", default=None)
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--preview-scale", type=float, default=0.5,
                    help="imshow 크기 스케일(0.1~1.0). 0.5면 절반 크기")
    ap.add_argument("--person-mode", type=str, default="largest", choices=["largest"],
                    help="사람 선택 방식 (현재 largest만 지원)")
    ap.add_argument("--min-area", type=float, default=15000.0,
                    help="bbox 최소 면적(픽셀^2). 이보다 작은 사람은 무시")
    args = ap.parse_args()

    model, classes, C, hp = load_ckpt(args.ckpt)
    win = int(hp.get("win",60)); features = hp.get("features","xyconf"); norm = hp.get("norm","image")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    yolo = YOLO(args.model, task='pose')

    src = 0 if args.video is None else args.video
    cap = cv2.VideoCapture(src)
    if args.video is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    buf = collections.deque(maxlen=win)
    last_kxy = None
    ema_prob = None; alpha = 2.0/float(args.smooth+1) if args.smooth>1 else 1.0

    t0 = time.time(); frames=0; fps=0.0
    scale = max(0.1, min(1.0, float(args.preview_scale)))

    while True:
        ok, frame = cap.read()
        if not ok: break
        h,w = frame.shape[:2]

        results = yolo.predict(source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                               device=args.device, verbose=False)
        res = results[0]

        pid = choose_person_largest(res, min_area=args.min_area)

        if pid is None:
            cv2.putText(frame, "No valid person (by size).", (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
            cv2.putText(frame, f"Tip: lower --min-area (now {args.min_area:.0f}) or move closer.", (15,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
            if scale != 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            cv2.imshow("Pose+TCN Inference (largest only)", frame)
            if (cv2.waitKey(1)&0xFF)==ord('q'): break
            continue

        if hasattr(res.keypoints, "xy"):
            kxy_np = res.keypoints.xy[pid].cpu().numpy()
        else:
            kxy_np = res.keypoints.data[pid][:,:2].cpu().numpy()
        if res.keypoints.conf is not None:
            kcf_np = res.keypoints.conf[pid].cpu().numpy()
        else:
            kcf_np = np.ones((KPTS,),dtype=np.float32)
        kxy = kxy_np.astype(np.float32); kcf = kcf_np.astype(np.float32)
        last_kxy = kxy.copy()

        if args.draw:
            if (res.boxes is not None) and (len(res.boxes)>pid):
                x,y,ww,hh = res.boxes.xywh[pid].cpu().numpy().tolist()
                x1,y1,x2,y2 = int(x-ww/2), int(y-hh/2), int(x+ww/2), int(y+hh/2)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"area={int(ww*hh)}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            draw_pose(frame, kxy, draw=True)

        kxy_n = normalize_xy(kxy, w, h, norm)
        if features == "xy":
            feat = kxy_n.reshape(-1)
        elif features == "xyconf":
            feat = np.concatenate([kxy_n.reshape(-1), kcf])
        else:
            raise SystemExit("features must be xy or xyconf")
        buf.append(feat.astype(np.float32))

        pred_name="(warming)"; pred_prob=0.0
        if len(buf)==win:
            X = np.stack(buf, axis=0)[None,...]
            Xc = torch.from_numpy(X.transpose(0,2,1)).float().to(device)
            with torch.no_grad():
                logits = model(Xc)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            if ema_prob is None: ema_prob = prob
            else: ema_prob = (1-alpha)*ema_prob + alpha*prob
            pidc = int(np.argmax(ema_prob)); pred_name = classes[pidc]; pred_prob = float(ema_prob[pidc])

        frames += 1
        if frames % 10 == 0:
            t1 = time.time(); fps = 10.0/(t1-t0); t0 = t1

        cv2.putText(frame, f"FPS: {fps:.1f}", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2)
        cv2.putText(frame, f"Pred: {pred_name} ({pred_prob:.2f})", (15,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,0),2)
        cv2.putText(frame, f"[Largest only | min_area={int(args.min_area)}]", (15,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,200,200),1)
        cv2.putText(frame, "[Q] Quit", (15,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,200,200),1)

        if scale != 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        cv2.imshow("Pose+TCN Inference (largest only)", frame)
        if (cv2.waitKey(1)&0xFF)==ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
