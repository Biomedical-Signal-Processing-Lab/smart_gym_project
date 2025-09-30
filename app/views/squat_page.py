# views/squat_page.py
import cv2, time
from collections import deque
from PySide6.QtWidgets import QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, Qt
from core.page_base import PageBase
from core.mp_manager import PoseProcessor
from ui.overlay_painter import draw_count_top_left
from ui.score_painter import ScoreOverlay

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
        self.lbl_status = QLabel("Status: - | Squat: 0")  

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

        # ============ 운동 정보 기록 ===================
        self.btn_end = QPushButton("운동 종료")
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_end)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.btn_end.clicked.connect(self._end_workout)

        self._session_started_ts = None
        self._score_sum = 0.0
        self._score_n   = 0

        # ==============================================
        self.fps_f = FPSMeter(); self.fps_s = FPSMeter()
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

        self._min_knee_in_phase = None
        self.score_overlay = ScoreOverlay(self)
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

        self.FRONT_DOWN_TH = 110.0   # 정면 down
        self.SIDE_DOWN_TH  = 110.0   # 측면 down
        self.UP_TH         = 160.0   # up 
        self.DEBOUNCE_N    = 3       # 연속 프레임 요구(노이즈 방지)

        self.state = "UP"            
        self.reps = 0
        self._down_frame = 0
        self._up_frame   = 0

        self._connected = False

    def on_enter(self, ctx):
        self.ctx = ctx
        self.score_overlay.setGeometry(self.rect()) 
        self.score_overlay.raise_()

        if not self._connected:
            self.btn_f_start.clicked.connect(lambda: self.ctx.cam.start("front"))
            self.btn_f_stop.clicked.connect(lambda: self.ctx.cam.stop("front"))
            self.btn_s_start.clicked.connect(lambda: self.ctx.cam.start("side"))
            self.btn_s_stop.clicked.connect(lambda: self.ctx.cam.stop("side"))
            self._connected = True

        self._session_started_ts = time.time()
        self._score_sum = 0.0
        self._score_n   = 0
        self.state = "UP"; self.reps = 0; self._down_frame= 0; self._up_frame = 0

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

    # ---------------- 판별 ----------------
    def _current_min_knee(self, knees_front, knees_side):
        vals = []
        if knees_front is not None: vals += list(knees_front)
        if knees_side  is not None: vals += list(knees_side)
        return min(vals) if vals else None

    def _knee_color_by_angle(self, ang: float) -> QColor:
        if ang <= 46:   return QColor(0, 128, 255)   # 파란색
        if ang <= 48:   return QColor(0, 200, 0)     # 녹색
        if ang <= 50:   return QColor(255, 255, 255) # 흰색
        if ang <= 55:   return QColor(255, 140, 0)   # 주황
        return QColor(255, 0, 0)                     # 빨강

    def _score_by_angle(self, ang: float) -> int: # 45~60 -> 100점 ~ 0점
        a0, s0 = 46.0, 100.0
        a1, s1 = 60.0, 0.0
        t = (ang - a0) / (a1 - a0)
        t = max(0.0, min(1.0, t))
        return int(round((1.0 - t) * s0 + t * s1))

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

        knees_f = self._knees_from_meta(mf) 
        knees_s = self._knees_from_meta(ms)

        is_down_now = self._is_down(knees_f, knees_s)
        is_up_now   = self._is_up(knees_f, knees_s)

        if is_down_now:
            self._down_frame += 1
            self._up_frame = 0
        elif is_up_now:
            self._up_frame += 1
            self._down_frame = 0
        else:
            self._down_frame = 0
            self._up_frame = 0

        cur_min = self._current_min_knee(knees_f, knees_s)
        if self.state == "DOWN" and cur_min is not None:
            if self._min_knee_in_phase is None:
                self._min_knee_in_phase = cur_min
            else:
                self._min_knee_in_phase = min(self._min_knee_in_phase, cur_min)

        if self.state == "UP":
            if self._down_frame >= self.DEBOUNCE_N:
                self.state = "DOWN"
                self._min_knee_in_phase = None   
        else:
            if self._up_frame >= self.DEBOUNCE_N:
                self.state = "UP"
                self.reps += 1

                ang_for_color = self._min_knee_in_phase if self._min_knee_in_phase is not None else (cur_min or 180.0)
                color = self._knee_color_by_angle(ang_for_color)
                score = self._score_by_angle(ang_for_color)     
                self._score_sum += float(score)
                self._score_n   += 1
                self.score_overlay.show_score(str(score), 100, text_qcolor=color)
                self._min_knee_in_phase = None

        def fmt_pair(p):
            if p is None: return "- / -"
            return f"{p[0]:.1f}° / {p[1]:.1f}°"
        s = (f"Status: {self.state} | Squat: {self.reps} | "
            f"front knees: {fmt_pair(knees_f)} | side knees: {fmt_pair(knees_s)}")
        self.lbl_status.setText(s)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "score_overlay") and self.score_overlay is not None:
            self.score_overlay.setGeometry(self.rect())

    def _end_workout(self):
        ended_at = time.time()
        duration_sec = 0.0 if self._session_started_ts is None else (ended_at - self._session_started_ts)
        avg_score = (self._score_sum / self._score_n) if self._score_n > 0 else 0.0

        summary = {
            "exercise": "squat",
            "reps": int(self.reps),
            "avg_score": round(avg_score, 1),
            "duration_sec": int(duration_sec),
            "started_at": self._session_started_ts,
            "ended_at": ended_at,
        }
        try:
            if hasattr(self.ctx, "save_workout_session"):
                self.ctx.save_workout_session(summary)
        except Exception as e:
            print(f"[WARN] workout save failed: {e}")

        self.timer.stop()
        try:
            self.ctx.cam.stop("front"); self.ctx.cam.stop("side")
        except Exception:
            pass

        if hasattr(self.ctx, "goto_summary"):
            self.ctx.goto_summary(summary)

    def _update(self, name, video_label, info_label, fps_meter):
        frame = self.ctx.cam.frame(name)
        if frame is None: return
        fps_meter.tick()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape

        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
        draw_count_top_left(qimg, self.reps, "SQUAT")
        video_label.setPixmap(QPixmap.fromImage(qimg).scaled(video_label.width(), video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        meta = self.ctx.cam.meta(name) or {}
        fps  = fps_meter.fps()
        def fmt(x, f): return "-" if x is None else f.format(x)
        txt = (f"FPS {fps:4.1f} | Hip y: {fmt(meta.get('hip_y_px'), '{:.0f}px')} "
               f"({fmt(meta.get('hip_y_norm'), '{:.3f}')}) | Knee L/R: "
               f"{fmt(meta.get('knee_l_deg'), '{:5.1f}')}° / {fmt(meta.get('knee_r_deg'), '{:5.1f}')}°")
        info_label.setText(txt)
