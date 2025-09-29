# views/squat_page.py
import cv2, time
from collections import deque
from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox, QSizePolicy, QSplitter
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPainterPath, QPen
from PySide6.QtCore import QTimer, Qt
from core.page_base import PageBase
from mp_manager import PoseProcessor
from ui.overlay_painter import draw_count_top_left, TextStyle

class FPSMeter:
    def __init__(self, maxlen=30): self.ts=deque(maxlen=maxlen)
    def tick(self): self.ts.append(time.time())
    def fps(self):
        if len(self.ts)<2: return 0.0
        dt=self.ts[-1]-self.ts[0]; return 0.0 if dt<=0 else (len(self.ts)-1)/dt

class SquatPage(PageBase):
    def __init__(self):
        super().__init__()
        self.proc_front = PoseProcessor(model_complexity=1, name="pose_front")
        self.proc_side  = PoseProcessor(model_complexity=1, name="pose_side")
        self.skeleton_front_on = True; self.skeleton_side_on = True

        self.lbl_front = QLabel("front: no frame"); self.lbl_front.setMinimumSize(300,300)
        self.lbl_side  = QLabel("side:  no frame"); self.lbl_side.setMinimumSize(300,300)
        self.info_f = QLabel("front info: -"); self.info_s = QLabel("side info: -")
        self.lbl_status = QLabel("Status: - | Count: 0")  

        self.btn_f_start = QPushButton("Start front"); self.btn_f_stop = QPushButton("Stop front")
        self.btn_s_start = QPushButton("Start side");  self.btn_s_stop = QPushButton("Stop side")

        self.chk_front = QCheckBox("Skeleton(front)"); self.chk_front.setChecked(True); self.chk_front.stateChanged.connect(self._toggle_front)
        self.chk_side  = QCheckBox("Skeleton(side)");  self.chk_side.setChecked(True);  self.chk_side.stateChanged.connect(self._toggle_side)

        g1=QGroupBox("front"); v1=QVBoxLayout(); v1.addWidget(self.lbl_front); v1.addWidget(self.info_f)
        h1=QHBoxLayout(); h1.addWidget(self.btn_f_start); h1.addWidget(self.btn_f_stop); h1.addWidget(self.chk_front); v1.addLayout(h1); g1.setLayout(v1)
        g2=QGroupBox("side");  v2=QVBoxLayout(); v2.addWidget(self.lbl_side);  v2.addWidget(self.info_s)
        h2=QHBoxLayout(); h2.addWidget(self.btn_s_start); h2.addWidget(self.btn_s_stop); h2.addWidget(self.chk_side);  v2.addLayout(h2); g2.setLayout(v2)
        root=QVBoxLayout(self)
        top=QHBoxLayout(); top.addWidget(g1); top.addWidget(g2)
        root.addLayout(top)
        root.addWidget(self.lbl_status) 

        self.fps_f = FPSMeter(); self.fps_s = FPSMeter()
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

        self.FRONT_DOWN_TH = 110.0   # 정면 down
        self.SIDE_DOWN_TH  = 110.0    # 측면 down
        self.UP_TH         = 160.0   # up 
        self.DEBOUNCE_N    = 3       # 연속 프레임 요구(노이즈 방지)

        self.state = "UP"            
        self.reps = 0
        self._down_streak = 0
        self._up_streak   = 0

        self._connected = False

    def on_enter(self, ctx):
        self.ctx = ctx
        if not self._connected:
            self.btn_f_start.clicked.connect(lambda: self.ctx.cam.start("front"))
            self.btn_f_stop.clicked.connect(lambda: self.ctx.cam.stop("front"))
            self.btn_s_start.clicked.connect(lambda: self.ctx.cam.start("side"))
            self.btn_s_stop.clicked.connect(lambda: self.ctx.cam.stop("side"))
            self._connected = True

        self.state = "UP"; self.reps = 0; self._down_streak = 0; self._up_streak = 0

        self._apply_processors()
        self.ctx.cam.start("front"); self.ctx.cam.start("side")
        self.timer.start(30)

    def on_leave(self, ctx):
        self.timer.stop()
        ctx.cam.stop("front"); ctx.cam.stop("side")

    # ---------------- UI/프로세서 ----------------
    def _toggle_front(self, _): self._apply_processors(one="front")
    def _toggle_side(self, _):  self._apply_processors(one="side")

    def _apply_processors(self, one=None):
        if one in (None,"front"):
            self.ctx.cam.set_process("front", self.proc_front if self.chk_front.isChecked() else None)
        if one in (None,"side"):
            self.ctx.cam.set_process("side",  self.proc_side  if self.chk_side.isChecked()  else None)

    # ---------------- 판별 유틸 (핵심 분리) ----------------
    @staticmethod
    def _knees_from_meta(m):
        kL, kR = m.get("knee_l_deg"), m.get("knee_r_deg")
        # 값이 하나라도 None이면 판정X
        if kL is None or kR is None:
            return None
        return float(kL), float(kR)

    def _is_down(self, knees_front, knees_side):
        if knees_front is None or knees_side is None:
            return False
        kLf, kRf = knees_front
        kLs, kRs = knees_side
        return (kLf < self.FRONT_DOWN_TH and kRf < self.FRONT_DOWN_TH and
                kLs < self.SIDE_DOWN_TH  and kRs < self.SIDE_DOWN_TH)

    def _is_up(self, knees_front, knees_side):
        if knees_front is None or knees_side is None:
            return False
        kLf, kRf = knees_front
        kLs, kRs = knees_side
        return (kLf >= self.UP_TH and kRf >= self.UP_TH and
                kLs >= self.UP_TH and kRs >= self.UP_TH)

    # ---------------- 메인 루프 ----------------
    def _tick(self):
        self._update("front", self.lbl_front, self.info_f, self.fps_f)
        self._update("side",  self.lbl_side,  self.info_s, self.fps_s)

        mf = self.ctx.cam.meta("front") or {}
        ms = self.ctx.cam.meta("side")  or {}

        knees_f = self._knees_from_meta(mf)  # (kL, kR) or None
        knees_s = self._knees_from_meta(ms)

        is_down_now = self._is_down(knees_f, knees_s)
        is_up_now   = self._is_up(knees_f, knees_s)

        if is_down_now:
            self._down_streak += 1
            self._up_streak = 0
        elif is_up_now:
            self._up_streak += 1
            self._down_streak = 0
        else:
            self._down_streak = 0
            self._up_streak   = 0

        if self.state == "UP":
            if self._down_streak >= self.DEBOUNCE_N:
                self.state = "DOWN"
        else:  
            if self._up_streak >= self.DEBOUNCE_N:
                self.state = "UP"
                self.reps += 1  # DOWN → UP 전환 시 1회 카운트

        def fmt_pair(p):
            if p is None: return "- / -"
            return f"{p[0]:.1f}° / {p[1]:.1f}°"
        s = (f"Status: {self.state} | Count: {self.reps} | "
             f"front knees: {fmt_pair(knees_f)} | side knees: {fmt_pair(knees_s)}")
        self.lbl_status.setText(s)

    def _update(self, name, video_label, info_label, fps_meter):
        frame = self.ctx.cam.frame(name)
        if frame is None: return
        fps_meter.tick()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape

        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
        draw_count_top_left(qimg, self.reps)
        video_label.setPixmap(QPixmap.fromImage(qimg).scaled(video_label.width(), video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        meta = self.ctx.cam.meta(name) or {}
        fps  = fps_meter.fps()
        def fmt(x, f): return "-" if x is None else f.format(x)
        txt = (f"FPS {fps:4.1f} | Hip y: {fmt(meta.get('hip_y_px'), '{:.0f}px')} "
               f"({fmt(meta.get('hip_y_norm'), '{:.3f}')}) | Knee L/R: "
               f"{fmt(meta.get('knee_l_deg'), '{:5.1f}')}° / {fmt(meta.get('knee_r_deg'), '{:5.1f}')}°")
        info_label.setText(txt)
