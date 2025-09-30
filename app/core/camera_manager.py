# core/camera_manager.py 
import cv2, threading, time
from copy import deepcopy

def _blank_meta():
    return {
        "ok": False,
        "ts": 0.0,
        "hip_y_px": None,
        "hip_y_norm": None,
        "knee_l_deg": None,
        "knee_r_deg": None,
    }

class CameraWorker(threading.Thread):
    def __init__(self, name, device_path, width, height, fps, process_fn=None):
        super().__init__(daemon=True)
        self.name = name
        self.device_path = device_path
        self.width, self.height, self.fps = width, height, fps
        self.process_fn = process_fn
        self._processor_close = getattr(process_fn, "close", None) if process_fn else None

        self._running = False
        self._cap = None
        self._lock = threading.Lock()
        self._latest = None
        self._latest_meta = _blank_meta()

    def _open(self):
        cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS,          self.fps)
        # MJPG 시도(지원 안 해도 무시)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        self._cap = cap

    def _close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def run(self):
        while True:
            if self._running and self._cap is None:
                self._open()

            if not self._running:
                self._close()
                time.sleep(0.05)
                continue

            if self._cap and self._cap.isOpened():
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue

                # 처리
                if self.process_fn:
                    try:
                        if hasattr(self.process_fn, "process"):
                            frame = self.process_fn.process(frame)
                        else:
                            frame = self.process_fn(frame)
                    except Exception:
                        pass

                meta = _blank_meta()
                try:
                    if self.process_fn:
                        get_meta = getattr(self.process_fn, "get_meta", None)
                        if callable(get_meta):
                            m = get_meta()
                            if isinstance(m, dict):
                                meta.update(m)  
                except Exception:
                    pass

                with self._lock:
                    self._latest = frame
                    self._latest_meta = meta  
            else:
                time.sleep(0.01)

    # 외부 API
    def start_capture(self): self._running = True
    def stop_capture(self):  self._running = False

    def get_latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def get_latest_meta(self):
        with self._lock:
            return deepcopy(self._latest_meta)

    def set_process(self, fn):
        with self._lock:
            self.process_fn = fn
            self._processor_close = getattr(fn, "close", None) if fn else None
            self._latest_meta = _blank_meta()

    def close_processor(self):
        if self._processor_close:
            try:
                self._processor_close()
            except Exception:
                pass
            self._processor_close = None

class CameraManager:
    def __init__(self, cam_map, width, height, fps):
        self.workers = {}
        for name, path in cam_map.items():
            w = CameraWorker(name, path, width, height, fps)
            w.start()
            self.workers[name] = w

    def names(self): return list(self.workers.keys())
    def start(self, name): self.workers[name].start_capture()
    def stop(self, name):  self.workers[name].stop_capture()
    def frame(self, name): return self.workers[name].get_latest()
    def meta(self, name):  return self.workers[name].get_latest_meta()
    def set_process(self, name, fn): self.workers[name].set_process(fn)
    def close_processors(self):
        for w in self.workers.values():
            w.close_processor()
