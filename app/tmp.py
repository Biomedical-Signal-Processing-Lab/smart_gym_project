#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 Pose + 각도 HUD (무릎/고관절/어깨/팔꿈치 + 정면 HipLine 고정판)
- HipLine을 '방향'의 영향을 제거한 0~90° 각도로 수정
  (왼쪽은 180°, 오른쪽은 2°처럼 튀는 현상 방지)
"""

from __future__ import annotations
import cv2, csv, time, json, argparse
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np

# === 스켈레톤(17 keypoints, COCO order) ===
SKELETON_EDGES = [
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (11,12),(5,11),(6,12),
    (11,13),(13,15),(12,14),(14,16),
    (0,1),(0,2),(1,3),(2,4)
]

def now_ms()->int: return int(time.time()*1000)
def ensure_dir(p:Path): p.mkdir(parents=True, exist_ok=True)
def put_text(img, text, org, scale=0.9, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_pose(img, kpts_xy: np.ndarray, draw_ids: bool=False):
    for i,j in SKELETON_EDGES:
        if np.all(kpts_xy[[i,j],:].min(axis=None) >= 0):
            p1 = tuple(map(int, kpts_xy[i])); p2 = tuple(map(int, kpts_xy[j]))
            cv2.line(img, p1, p2, (0,255,255), 2)
    for idx,(x,y) in enumerate(kpts_xy):
        if x>=0 and y>=0:
            cv2.circle(img,(int(x),int(y)),6,(255,0,255),-1)
            if draw_ids:
                cv2.putText(img, str(idx), (int(x)+6, int(y)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(img, str(idx), (int(x)+6, int(y)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def ort_has_cuda()->bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False

# ---------------- Angle Utils ----------------
def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC (at B) in degrees, range 0~180."""
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return np.nan
    cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def _ok(kxy, kcf, idxs, thr):
    return all((kxy[i,0]>=0 and kxy[i,1]>=0 and kcf[i] >= thr) for i in idxs)

def compute_joint_angles(kxy: np.ndarray, kcf: np.ndarray, conf_thr: float=0.2) -> Dict[str, float]:
    # indices
    LSh, RSh = 5, 6
    LEl, REl = 7, 8
    LWr, RWr = 9,10
    LHp, RHp = 11,12
    LKn, RKn = 13,14
    LAn, RAn = 15,16

    ang: Dict[str, float] = {}

    # Knee: Hip-Knee-Ankle
    if _ok(kxy,kcf,[LHp,LKn,LAn],conf_thr): ang["Knee(L)"] = _angle_deg(kxy[LHp], kxy[LKn], kxy[LAn])
    else:                                   ang["Knee(L)"] = np.nan
    if _ok(kxy,kcf,[RHp,RKn,RAn],conf_thr): ang["Knee(R)"] = _angle_deg(kxy[RHp], kxy[RKn], kxy[RAn])
    else:                                   ang["Knee(R)"] = np.nan

    # Hip: Shoulder-Hip-Knee
    if _ok(kxy,kcf,[LSh,LHp,LKn],conf_thr): ang["Hip(L)"]  = _angle_deg(kxy[LSh], kxy[LHp], kxy[LKn])
    else:                                   ang["Hip(L)"]  = np.nan
    if _ok(kxy,kcf,[RSh,RHp,RKn],conf_thr): ang["Hip(R)"]  = _angle_deg(kxy[RSh], kxy[RHp], kxy[RKn])
    else:                                   ang["Hip(R)"]  = np.nan

    # Shoulder: Hip-Shoulder-Elbow
    if _ok(kxy,kcf,[LHp,LSh,LEl],conf_thr): ang["Shoulder(L)"] = _angle_deg(kxy[LHp], kxy[LSh], kxy[LEl])
    else:                                   ang["Shoulder(L)"] = np.nan
    if _ok(kxy,kcf,[RHp,RSh,REl],conf_thr): ang["Shoulder(R)"] = _angle_deg(kxy[RHp], kxy[RSh], kxy[REl])
    else:                                   ang["Shoulder(R)"] = np.nan

    # Elbow: Shoulder-Elbow-Wrist
    if _ok(kxy,kcf,[LSh,LEl,LWr],conf_thr): ang["Elbow(L)"] = _angle_deg(kxy[LSh], kxy[LEl], kxy[LWr])
    else:                                   ang["Elbow(L)"] = np.nan
    if _ok(kxy,kcf,[RSh,REl,RWr],conf_thr): ang["Elbow(R)"] = _angle_deg(kxy[RSh], kxy[REl], kxy[RWr])
    else:                                   ang["Elbow(R)"] = np.nan

    # HipLine (front): direction-agnostic 0..90° relative to horizontal
    def _hip_line_angle(hip, knee):
        dx, dy = knee[0]-hip[0], knee[1]-hip[1]
        if abs(dx)<1e-6 and abs(dy)<1e-6: return np.nan
        return float(np.degrees(np.arctan2(abs(dy), abs(dx))))  # 0..90
    if _ok(kxy,kcf,[LHp,LKn],conf_thr): ang["HipLine(L)"] = _hip_line_angle(kxy[LHp], kxy[LKn])
    else:                               ang["HipLine(L)"] = np.nan
    if _ok(kxy,kcf,[RHp,RKn],conf_thr): ang["HipLine(R)"] = _hip_line_angle(kxy[RHp], kxy[RKn])
    else:                               ang["HipLine(R)"] = np.nan

    return ang

def draw_angle_hud(preview_img, angles: Dict[str, float], x: int=20, y0: int=240):
    put_text(preview_img, "Angles (deg):", (x, y0), 0.9, (255,255,0), 2)
    y = y0 + 30
    order = ["Hip(L)","Hip(R)","Knee(L)","Knee(R)","Shoulder(L)","Shoulder(R)","Elbow(L)","Elbow(R)","HipLine(L)","HipLine(R)"]
    for name in order:
        v = angles.get(name, np.nan)
        txt = f"{name:12s}: {'-' if np.isnan(v) else f'{v:6.1f}'}"
        put_text(preview_img, txt, (x, y), 0.85, (200,255,200) if not np.isnan(v) else (180,180,180), 2)
        y += 28

# ---------------- Inference Wrapper ----------------
class YOLOv8Pose:
    def __init__(self, model_path:str, imgsz:int=640, conf:float=0.25, iou:float=0.7,
                 device:str="0", min_person_conf:float=0.7):
        from ultralytics import YOLO
        self.model = YOLO(model_path, task="pose")
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.device = device if (device=="cpu" or ort_has_cuda()) else "cpu"
        self.min_person_conf = float(min_person_conf)
        try:
            import onnxruntime as ort
            print(f"[INFO] ORT {ort.__version__} providers={ort.get_available_providers()} device={self.device}")
        except Exception:
            pass

    def infer(self, bgr: np.ndarray):
        res = self.model.predict(source=bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=(0 if self.device!='cpu' else 'cpu'), verbose=False)[0]
        n = len(res.boxes) if res.boxes is not None else 0
        if n==0 or res.keypoints is None:
            return None

        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes,'conf') else np.zeros((n,),dtype=np.float32)
        keep = np.where(confs >= self.min_person_conf)[0]
        if keep.size == 0:
            return None

        best = keep[np.argmax(confs[keep])]
        kxy = res.keypoints.xy.cpu().numpy()[best]
        try: kcf = res.keypoints.conf.cpu().numpy()[best]
        except Exception: kcf = np.ones((17,), dtype=np.float32)
        return {'kpts_xy':kxy,'kpts_conf':kcf,'person_conf':float(confs[best])}

# ---------------- CSV/REC utils ----------------
def build_header()->List[str]:
    cols=["timestamp_ms","label","view","cam_id","width","height"]
    for i in range(17): cols += [f"kpt{i}_x",f"kpt{i}_y",f"kpt{i}_conf"]
    return cols

def row_from_pose(ts:int,label:str,view:str,cam_id:int,w:int,h:int,kxy:np.ndarray,kcf:np.ndarray):
    row=[ts,label,view,cam_id,w,h]
    for i in range(17): row += [float(kxy[i,0]), float(kxy[i,1]), float(kcf[i])]
    return row

def fourcc_from_str(s:str)->int:
    try: return cv2.VideoWriter_fourcc(*s)
    except Exception: return cv2.VideoWriter_fourcc(*'MJPG')

def parse_thresh(v: float) -> float:
    v = float(v)
    return v/100.0 if v>1.0 else v

class Recorder:
    def __init__(self, session_dir: Path, base_name: str, fourcc: str, fps: float, size_src: str='preview'):
        self.session_dir = session_dir
        self.base_name = base_name
        self.fourcc = fourcc
        self.fps = fps
        self.size_src = size_src
        self.writer: Optional[cv2.VideoWriter] = None
        self.part = 0
        self.w = None
        self.h = None
    def _next_path(self)->Path:
        self.part += 1
        ext = '.mp4' if self.fourcc.lower()=='mp4v' else '.avi'
        return self.session_dir / f"{self.base_name}_part{self.part:02d}{ext}"
    def start(self, frame_for_size):
        if self.writer is not None: return
        self.h, self.w = frame_for_size.shape[:2]
        path = self._next_path()
        writer = cv2.VideoWriter(str(path), fourcc_from_str(self.fourcc.upper()), self.fps, (self.w, self.h))
        if not writer.isOpened():
            alt_path = path.with_suffix('.avi')
            writer = cv2.VideoWriter(str(alt_path), fourcc_from_str('MJPG'), self.fps, (self.w, self.h))
            if not writer.isOpened():
                raise RuntimeError("VideoWriter open failed: mp4v/MJPG codec check")
            path = alt_path
        self.writer = writer
        print(f"[REC] start -> {path} ({self.w}x{self.h}@{self.fps:.1f})")
    def stop(self):
        if self.writer is not None:
            self.writer.release(); self.writer = None
            print("[REC] stop")
    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

# ---------------- Main ----------------
def main():
    ap=argparse.ArgumentParser(description="YOLOv8 Pose 각도 HUD + 수집/녹화 (HipLine 고정판)")
    ap.add_argument('--model',type=str,default='yolov8m_pose.onnx')
    ap.add_argument('--cam-id',type=int,default=0)
    ap.add_argument('--width',type=int,default=1920)
    ap.add_argument('--height',type=int,default=1080)
    ap.add_argument('--fps',type=int,default=30)
    ap.add_argument('--imgsz',type=int,default=640)
    ap.add_argument('--conf',type=float,default=0.25)
    ap.add_argument('--iou',type=float,default=0.7)
    ap.add_argument('--device',type=str,default='0')
    ap.add_argument('--min-person-conf',type=float,default=0.7, help='사람(box) 신뢰도 (0~1 또는 70=0.70)')
    ap.add_argument('--min-kpt-conf',type=float,default=0.2, help='각도 계산에 사용할 키포인트 최소 conf')
    ap.add_argument('--labels',type=str,default='idle,squat,lunge,pushup,plank')
    ap.add_argument('--out-dir',type=str,default='./dataset')
    ap.add_argument('--view',type=str,default='front')
    ap.add_argument('--downsample',type=int,default=1)
    ap.add_argument('--fourcc',type=str,default='MJPG')
    ap.add_argument('--preview-scale',type=float,default=0.8)
    ap.add_argument('--imshow-skip',dest='imshow_skip',type=int,default=1)
    ap.add_argument('--hud-only',dest='hud_only',action='store_true')
    ap.add_argument('--draw-ids',dest='draw_ids',action='store_true')
    ap.add_argument('--rec-fourcc',type=str,default='mp4v')
    ap.add_argument('--rec-size',type=str,default='preview', choices=['preview','full'])
    args=ap.parse_args()

    labels_list=[s.strip() for s in args.labels.split(',') if s.strip()]
    if not (1<=len(labels_list)<=9): raise SystemExit('라벨 개수는 1~9 사이여야 합니다.')
    min_person_conf = parse_thresh(args.min_person_conf)

    session_name=time.strftime('%Y%m%d_%H_%M_%S')
    session_dir=Path(args.out_dir)/f'session_{session_name}'
    ensure_dir(session_dir)
    with open(session_dir/'session_meta.json','w',encoding='utf-8') as f:
        json.dump({'model':args.model,'imgsz':args.imgsz,'labels':labels_list,
                   'key_mapping':{str(i+1):labels_list[i] for i in range(len(labels_list))},
                   'camera':{'cam_id':args.cam_id,'width':args.width,'height':args.height,'fps':args.fps,'view':args.view},
                   'device':args.device, 'min_person_conf':min_person_conf,
                   'min_kpt_conf':args.min_kpt_conf,
                   'recording':{'fourcc':args.rec_fourcc,'size':args.rec_size,'fps':args.fps}}, f, ensure_ascii=False, indent=2)

    csv_path=session_dir/'pose_data.csv'
    fcsv=open(csv_path,'w',newline='',encoding='utf-8'); writer=csv.writer(fcsv); writer.writerow(build_header())

    cap=cv2.VideoCapture(args.cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    cap.set(cv2.CAP_PROP_FPS,args.fps)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc_from_str(args.fourcc))

    if not cap.isOpened():
        fcsv.close(); raise RuntimeError(f'카메라({args.cam_id})를 열 수 없습니다.')

    pose_model=YOLOv8Pose(args.model,imgsz=args.imgsz,conf=args.conf,iou=args.iou,
                          device=args.device, min_person_conf=min_person_conf)

    rec_on=False
    video_on=False
    current_label=labels_list[0]
    saved_frames=0; frame_count=0
    t0=time.time(); fps=0.0; fps_every=10

    recorder = Recorder(session_dir, base_name='session_video',
                        fourcc=args.rec_fourcc, fps=float(args.fps), size_src=args.rec_size)

    win='Pose Data Collector (Angles HUD - HipLine fixed)'
    cv2.namedWindow(win,cv2.WINDOW_NORMAL); cv2.resizeWindow(win,1280,720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                blank = np.zeros((480,640,3),dtype=np.uint8)
                put_text(blank, 'Camera Read Failed', (30,50), 1.0, (0,0,255), 2)
                cv2.imshow(win, blank)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            result = pose_model.infer(frame)
            h, w = frame.shape[:2]

            # 스켈레톤 드로잉
            if result is not None and not args.hud_only:
                draw_pose(frame, result['kpts_xy'], draw_ids=args.draw_ids)

            # 프리뷰
            preview = frame
            if 0.1 <= args.preview_scale < 1.0:
                preview = cv2.resize(frame, None, fx=args.preview_scale, fy=args.preview_scale, interpolation=cv2.INTER_LINEAR)
                ph, pw = preview.shape[:2]
            else:
                ph, pw = h, w

            # CSV 저장
            if rec_on and result is not None and (frame_count % args.downsample == 0):
                writer.writerow(row_from_pose(now_ms(), current_label, args.view, args.cam_id, w, h,
                                              result['kpts_xy'], result['kpts_conf']))
                saved_frames += 1

            # FPS
            frame_count += 1
            if frame_count % fps_every == 0:
                t1 = time.time()
                fps = fps_every / (t1 - t0 + 1e-9)
                t0 = t1

            # HUD 텍스트
            put_text(preview, f"Label: {current_label}", (20,40), 0.9, (255,255,0), 2)
            put_text(preview, f"CSV: {'ON' if rec_on else 'OFF'}  saved={saved_frames}", (20,80), 0.9,
                     (0,255,0) if rec_on else (200,200,200), 2)
            put_text(preview, f"VID: {'ON' if video_on else 'OFF'}", (20,120), 0.9,
                     (0,200,255) if video_on else (200,200,200), 2)
            put_text(preview, f"FPS: {fps:4.1f}", (20,160), 0.9, (0,255,255), 2)
            put_text(preview, f"minPersonConf: {min_person_conf:.2f}  minKptConf: {args.min_kpt_conf:.2f}",
                     (20,200), 0.8, (200,255,200), 2)

            # 각도 계산 + 표시
            if result is not None:
                angles = compute_joint_angles(result['kpts_xy'], result['kpts_conf'], conf_thr=args.min_kpt_conf)
                draw_angle_hud(preview, angles, x=20, y0=240)

            # 프리뷰 표시 주기
            if args.imshow_skip <= 1 or (frame_count % args.imshow_skip == 0):
                cv2.imshow(win, preview)

            # 녹화(미리보기/풀프레임)
            if video_on:
                frame_to_write = preview if args.rec_size=='preview' else frame
                recorder.start(frame_to_write)
                recorder.write(frame_to_write)
            else:
                recorder.stop()

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                recorder.stop()
                break
            elif key == ord('s'):
                rec_on = True
            elif key == ord('t'):
                rec_on = False
            elif key == ord('r'):
                video_on = True
            elif key == ord('e'):
                video_on = False
            elif key in [ord(str(i)) for i in range(1,10)]:
                idx = key - ord('1')
                if 0 <= idx < len(labels_list):
                    current_label = labels_list[idx]

    finally:
        cap.release(); recorder.stop(); fcsv.close(); cv2.destroyAllWindows()
        print(f"[DONE] Saved {saved_frames} frames -> {csv_path}")

if __name__ == "__main__":
    main()