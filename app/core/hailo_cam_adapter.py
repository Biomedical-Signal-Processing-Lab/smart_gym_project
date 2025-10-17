# core/hailo_cam_adapter.py
# ONNX (YOLOv8-Pose + TCN) 어댑터
# - Public API: HailoCamAdapter.start/stop/frame/people/meta
# - 화면 디버그 출력 없음. 콘솔 로깅만 사용.
# - frame()은 이제 **RGB** 프레임을 반환합니다.

from __future__ import annotations
import os, json, time, math, threading, collections, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import onnxruntime as ort

# --- 로거 준비 (상위에서 설정 안 했으면 기본 핸들러 구성) ---
LOG = logging.getLogger("HailoCamAdapter")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ultralytics는 ONNX pose 실행 편의용
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics가 필요합니다. pip install ultralytics") from e

# -------------------- Config --------------------
POSE_ONNX  = os.environ.get("POSE_ONNX", "models/yolov8m_pose.onnx")
TCN_ONNX   = os.environ.get("TCN_ONNX",  "models/tcn.onnx")
META_JSON  = os.environ.get("TCN_JSON",  "models/tcn.json")

IMG_SIZE   = int(os.environ.get("IMG_SIZE", "640"))
POSE_CONF  = float(os.environ.get("POSE_CONF", "0.25"))
POSE_IOU   = float(os.environ.get("POSE_IOU",  "0.50"))
MIN_PERSON_CONF = float(os.environ.get("MIN_PERSON_CONF", "0.70"))
SMOOTH_K   = int(os.environ.get("SMOOTH_K", "5"))  # EMA window for class probs

def _choose_source() -> int | str:
    s = os.environ.get("SOURCE", "").strip()
    if s:
        return int(s) if s.isdigit() else s
    vids = sorted([p for p in Path(".").glob("*") if p.suffix.lower() in (".mp4",".avi",".mov",".mkv",".webm")])
    return str(vids[0]) if vids else 0

# COCO17 index
KPTS = 17
L_HIP,L_KNEE,L_ANK = 11, 13, 15
R_HIP,R_KNEE,R_ANK = 12, 14, 16

# -------------------- Small math helpers --------------------
def _angle(a, b, c) -> Optional[float]:
    if a is None or b is None or c is None: return None
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax-bx, ay-by], dtype=np.float32)
    v2 = np.array([cx-bx, cy-by], dtype=np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return None
    cosv = float(np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))

def _normalize_xy(kxy, w, h):
    out = kxy.astype(np.float32).copy()
    out[:,0] /= max(1.0, float(w))
    out[:,1] /= max(1.0, float(h))
    return out

def _softmax_np(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e / np.sum(e, axis=-1, keepdims=True)

def _pick_providers():
    avail = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail] or ["CPUExecutionProvider"]

def _inspect_tcn_io(sess):
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)  # [1, C, T] or dynamic
    C = int(shape[1])
    T = None
    try:
        if isinstance(shape[2], (int, np.integer)): T = int(shape[2])
    except Exception:
        T = None
    return C, T

def _choose_largest_person(result, min_conf=0.7):
    if result.boxes is None or len(result.boxes)==0 or result.keypoints is None:
        return None
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
    xyxy  = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else None
    if confs is None or xyxy is None:
        return None
    best_i, best_area = -1, -1.0
    for i,c in enumerate(confs):
        if c < min_conf: continue
        x1,y1,x2,y2 = xyxy[i]
        area = max(0,x2-x1)*max(0,y2-y1)
        if area > best_area:
            best_area, best_i = area, i
    return best_i if best_i>=0 else None

# ============================================================
# Public adapter
# ============================================================
class HailoCamAdapter:
    def __init__(self, conf_thr: float = 0.65, stride: int = 1,
                 onnx_path: str | None = None, json_path: str | None = None):
        # conf_thr: keypoint accept & person min conf
        self.conf_thr = conf_thr if conf_thr <= 1.5 else conf_thr/100.0
        self.stride   = int(max(1, stride))

        self._lock = threading.Lock()
        self._frame_rgb: Optional[np.ndarray] = None   # NOTE: stores RGB
        self._people: List[Dict[str, Any]] = []
        self._cls: Optional[Dict[str, Any]] = None
        self._size: Tuple[int,int] = (640, 480)
        self._running = False
        self._th: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # error relay for UI (필요 시 UI에서 읽을 수 있음)
        self._last_error: Optional[str] = None

        # model paths
        self._tcn_onnx = onnx_path or TCN_ONNX
        self._tcn_json = json_path or META_JSON
        self._pose_onnx = POSE_ONNX

    # ---------- lifecycle ----------
    def start(self):
        if self._running:
            return
        self._stop_flag.clear()
        self._th = threading.Thread(target=self._worker, name="pose_tcn_worker", daemon=True)
        self._th.start()
        self._running = True
        LOG.info("adapter started")

    def stop(self):
        if not self._running:
            return
        self._stop_flag.set()
        if self._th is not None:
            self._th.join(timeout=2.0)
        self._running = False
        LOG.info("adapter stopped")

    # ---------- pullers (public API) ----------
    def frame(self) -> Optional[np.ndarray]:
        # 반환은 **RGB** 프레임
        with self._lock:
            return None if self._frame_rgb is None else self._frame_rgb.copy()

    def people(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._people)

    def meta(self) -> Dict[str, Any]:
        with self._lock:
            ok = bool(self._people)
            w, h = self._size
            label = self._cls.get("label") if isinstance(self._cls, dict) else None
            score = float(self._cls.get("score")) if isinstance(self._cls, dict) and "score" in self._cls else None

            knees = (None, None)
            if self._people:
                p = self._people[0]
                pts = p.get("kpt", [])
                def pt(idx):
                    if idx >= len(pts): return None
                    x, y = int(pts[idx][0]), int(pts[idx][1])
                    c = float(pts[idx][2]) if len(pts[idx]) >= 3 else 1.0
                    return (x,y) if c >= self.conf_thr else None
                l_ang = _angle(pt(L_HIP), pt(L_KNEE), pt(L_ANK))
                r_ang = _angle(pt(R_HIP), pt(R_KNEE), pt(R_ANK))
                knees = (l_ang, r_ang)

            return {
                "ok": ok,
                "src_w": w, "src_h": h,
                "label": label,
                "score": score,          # 0.0~1.0
                "knee_l_deg": knees[0],
                "knee_r_deg": knees[1],
            }

    # ---------- worker ----------
    def _worker(self):
        LOG.info("worker: start")
        # --- load meta ---
        meta_path = Path(self._tcn_json)
        if not meta_path.exists():
            self._last_error = f"missing meta json: {self._tcn_json}"
            LOG.error(self._last_error)
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = meta.get("classes", [])
        ncls = len(classes)
        hp = meta.get("hparams", {})
        win_meta = int(hp.get("win", 60))
        LOG.info("meta loaded: ncls=%d, win_meta=%d", ncls, win_meta)

        # --- TCN (GPU 실패 시 CPU 폴백) ---
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            providers = _pick_providers()
            LOG.info("providers available: %s", providers)
            tcn_sess = ort.InferenceSession(self._tcn_onnx, sess_options=so, providers=providers)
            LOG.info("TCN session ok: %s", self._tcn_onnx)
        except Exception as e:
            self._last_error = f"TCN GPU init failed: {e}"
            LOG.exception(self._last_error)
            try:
                tcn_sess = ort.InferenceSession(self._tcn_onnx, sess_options=so, providers=["CPUExecutionProvider"])
                LOG.info("TCN session fallback to CPU ok")
            except Exception as e2:
                self._last_error = f"TCN CPU init failed: {e2}"
                LOG.exception(self._last_error)
                return

        C, T_fixed = _inspect_tcn_io(tcn_sess)
        win = T_fixed if T_fixed is not None else win_meta
        LOG.info("TCN IO: C=%d, win=%s", C, win)

        # --- YOLO Pose ---
        if not Path(self._pose_onnx).exists():
            self._last_error = f"missing pose onnx: {self._pose_onnx}"
            LOG.error(self._last_error)
            return
        try:
            yolo = YOLO(self._pose_onnx, task="pose")
            LOG.info("YOLO pose ok: %s", self._pose_onnx)
        except Exception as e:
            self._last_error = f"YOLO init failed: {e}"
            LOG.exception(self._last_error)
            return

        # --- source open ---
        src = _choose_source()
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self._last_error = f"Failed to open source: {src}"
            LOG.error(self._last_error)
            return
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit() and int(src) == 0):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        LOG.info("source opened: %r", src)
        self._last_error = "worker: loop running"

        # buffers
        buf = collections.deque(maxlen=win)
        ema_prob = None
        alpha = 2.0/float(max(1, SMOOTH_K)+1) if SMOOTH_K>0 else 1.0

        last_pred = 0
        pred_prob = 0.0

        # periodic log timer
        last_log_ts = time.time()

        # main loop
        stride_cnt = 0
        while not self._stop_flag.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            h, w = frame_bgr.shape[:2]

            people_list: List[Dict[str, Any]] = []

            # run pose every "stride" frames
            do_pose = (stride_cnt % max(1, self.stride) == 0)

            if do_pose:
                try:
                    # YOLO는 BGR frame을 써도 내부에서 처리 가능
                    res = yolo.predict(source=frame_bgr, imgsz=IMG_SIZE, conf=POSE_CONF, iou=POSE_IOU, verbose=False, device=None)[0]
                    pid = _choose_largest_person(res, min_conf=max(self.conf_thr, MIN_PERSON_CONF))
                    if pid is not None:
                        kxy = res.keypoints.xy[pid].cpu().numpy().astype(np.float32)  # (17,2)
                        kcf = (res.keypoints.conf[pid].cpu().numpy().astype(np.float32)
                               if res.keypoints.conf is not None else np.ones((KPTS,), dtype=np.float32))
                        people_list.append({
                            "kpt": np.concatenate([kxy, kcf[:,None]], axis=1).tolist(),  # [[x,y,c], ...] length 17
                            "conf": float(res.boxes.conf[pid].item()) if res.boxes.conf is not None else 1.0,
                        })
                        # feature for TCN: 34 (xy) normalized + 17 conf = 51
                        kxy_n = _normalize_xy(kxy, w, h)
                        feat = np.concatenate([kxy_n.reshape(-1), kcf], axis=0).astype(np.float32)
                    else:
                        # no person → zeros
                        feat = np.zeros((C,), dtype=np.float32)

                    buf.append(feat)

                    # TCN when buffer is full
                    if len(buf) == win:
                        X = np.stack(buf, axis=1).astype(np.float32)  # (C,T)
                        X = X[None, ...]  # (1,C,T)
                        ort_inputs = {tcn_sess.get_inputs()[0].name: X}
                        logits = tcn_sess.run(None, ort_inputs)[0]  # (1,ncls)
                        prob = _softmax_np(logits)[0]
                        ema_prob = prob if ema_prob is None else (1-alpha)*ema_prob + alpha*prob
                        last_pred = int(np.argmax(ema_prob))
                        pred_prob = float(ema_prob[last_pred])
                except Exception as e:
                    self._last_error = f"infer step failed: {e}"
                    LOG.exception(self._last_error)

            stride_cnt += 1

            # --- 여기서 RGB로 변환 후 공유 상태에 저장 ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            with self._lock:
                self._frame_rgb = frame_rgb.copy()     # NOTE: stores RGB
                self._people = people_list
                self._cls = {"label": classes[last_pred] if 0 <= last_pred < ncls else None,
                             "score": pred_prob}
                self._size = (w, h)

            # --- 1초마다 요약 로그 ---
            now = time.time()
            if now - last_log_ts >= 1.0:
                last_log_ts = now
                try:
                    lbl = classes[last_pred] if 0 <= last_pred < ncls else None
                    LOG.info("frame=%dx%d people=%d label=%s score=%.2f",
                             w, h, len(people_list), lbl, pred_prob)
                except Exception:
                    pass

        cap.release()
        LOG.info("worker: exit")
