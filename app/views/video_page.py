# views/video_page.py
import cv2, time
from collections import deque
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer
from core.page_base import PageBase
from mp_manager import PoseProcessor

class FPSMeter:
    def __init__(self, maxlen=30): self.ts=deque(maxlen=maxlen)
    def tick(self): self.ts.append(time.time())
    def fps(self):
        if len(self.ts)<2: return 0.0
        dt=self.ts[-1]-self.ts[0]; return 0.0 if dt<=0 else (len(self.ts)-1)/dt

class VideoPage(PageBase):
    def __init__(self):
        super().__init__()
        self.proc_front = PoseProcessor(model_complexity=1, name="pose_front")
        self.proc_side  = PoseProcessor(model_complexity=1, name="pose_side")
        self.skeleton_front_on = True; self.skeleton_side_on = True

        self.lbl_front = QLabel("front: no frame"); self.lbl_front.setMinimumSize(480,270)
        self.lbl_side  = QLabel("side:  no frame"); self.lbl_side.setMinimumSize(480,270)
        self.info_f = QLabel("front info: -"); self.info_s = QLabel("side info: -")

        self.btn_f_start = QPushButton("Start front"); self.btn_f_stop = QPushButton("Stop front")
        self.btn_s_start = QPushButton("Start side");  self.btn_s_stop = QPushButton("Stop side")

        self.chk_front = QCheckBox("Skeleton(front)"); self.chk_front.setChecked(True); self.chk_front.stateChanged.connect(self._toggle_front)
        self.chk_side  = QCheckBox("Skeleton(side)");  self.chk_side.setChecked(True);  self.chk_side.stateChanged.connect(self._toggle_side)

        g1=QGroupBox("front"); v1=QVBoxLayout(); v1.addWidget(self.lbl_front); v1.addWidget(self.info_f)
        h1=QHBoxLayout(); h1.addWidget(self.btn_f_start); h1.addWidget(self.btn_f_stop); h1.addWidget(self.chk_front); v1.addLayout(h1); g1.setLayout(v1)
        g2=QGroupBox("side");  v2=QVBoxLayout(); v2.addWidget(self.lbl_side);  v2.addWidget(self.info_s)
        h2=QHBoxLayout(); h2.addWidget(self.btn_s_start); h2.addWidget(self.btn_s_stop); h2.addWidget(self.chk_side);  v2.addLayout(h2); g2.setLayout(v2)
        root=QHBoxLayout(self); root.addWidget(g1); root.addWidget(g2)

        self.fps_f = FPSMeter(); self.fps_s = FPSMeter()
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

    def on_enter(self, ctx):
        self.ctx = ctx
        # 버튼 → 실제 캡처 제어
        self.btn_f_start.clicked.connect(lambda: self.ctx.cam.start("front"))
        self.btn_f_stop.clicked.connect(lambda: self.ctx.cam.stop("front"))
        self.btn_s_start.clicked.connect(lambda: self.ctx.cam.start("side"))
        self.btn_s_stop.clicked.connect(lambda: self.ctx.cam.stop("side"))

        # 초기 프로세서 장착
        self._apply_processors()
        self.ctx.cam.start("front"); self.ctx.cam.start("side")

        self.timer.start(30)

    def on_leave(self, ctx):
        self.timer.stop()
        ctx.cam.stop("front"); ctx.cam.stop("side")

    def _toggle_front(self, _): self._apply_processors(one="front")
    def _toggle_side(self, _):  self._apply_processors(one="side")

    def _apply_processors(self, one=None):
        if one in (None,"front"):
            self.ctx.cam.set_process("front", self.proc_front if self.chk_front.isChecked() else None)
        if one in (None,"side"):
            self.ctx.cam.set_process("side",  self.proc_side  if self.chk_side.isChecked()  else None)

    def _tick(self):
        self._update("front", self.lbl_front, self.info_f, self.fps_f)
        self._update("side",  self.lbl_side,  self.info_s, self.fps_s)

    def _update(self, name, video_label, info_label, fps_meter):
        frame = self.ctx.cam.frame(name)
        if frame is None: return
        fps_meter.tick()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        video_label.setPixmap(QPixmap.fromImage(qimg).scaled(video_label.width(), video_label.height()))

        meta = self.ctx.cam.meta(name) or {}
        fps  = fps_meter.fps()
        def fmt(x, f): return "-" if x is None else f.format(x)
        txt = (f"FPS {fps:4.1f} | Hip y: {fmt(meta.get('hip_y_px'), '{:.0f}px')} "
               f"({fmt(meta.get('hip_y_norm'), '{:.3f}')}) | Knee L/R: "
               f"{fmt(meta.get('knee_l_deg'), '{:5.1f}')}° / {fmt(meta.get('knee_r_deg'), '{:5.1f}')}°")
        info_label.setText(txt)
