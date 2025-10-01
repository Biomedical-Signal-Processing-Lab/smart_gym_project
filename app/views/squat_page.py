# views/squat_page.py
import cv2, time
from PySide6.QtWidgets import QLabel, QPushButton, QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, Qt
from core.page_base import PageBase
from core.mp_manager import PoseProcessor
from ui.overlay_painter import draw_count_top_left
from ui.score_painter import ScoreOverlay

class SquatPage(PageBase):
    def __init__(self):
        super().__init__()

        # ----- 단일 카메라용 Pose Processor -----
        self.proc = PoseProcessor(model_complexity=1, name="pose_camera")
        self.skeleton_on = True

        # ----- UI -----
        self.lbl_camera = QLabel("camera: no frame"); self.lbl_camera.setMinimumSize(400, 400)
        self.info = QLabel("camera info: -")
        self.lbl_status = QLabel("Status: - | Squat: 0")

        self.btn_start = QPushButton("Start camera")
        self.btn_stop  = QPushButton("Stop camera")

        self.chk_skel = QCheckBox("Skeleton(camera)")
        self.chk_skel.setChecked(True)
        self.chk_skel.stateChanged.connect(self._toggle_skeleton)

        g = QGroupBox("camera")
        v = QVBoxLayout()
        v.addWidget(self.lbl_camera)
        v.addWidget(self.info)
        h = QHBoxLayout()
        h.addWidget(self.btn_start)
        h.addWidget(self.btn_stop)
        h.addWidget(self.chk_skel)
        v.addLayout(h)
        g.setLayout(v)

        root = QVBoxLayout(self)
        root.addWidget(g)
        root.addWidget(self.lbl_status)

        # ============ 운동 정보 기록 ===================
        self.btn_end = QPushButton("운동 종료")
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_end)
        btn_row.addStretch(1)
        root.addLayout(btn_row)
        self.btn_end.clicked.connect(self._end_workout)

        # ----- 세션/스코어 상태 -----
        self._session_started_ts = None
        self._score_sum = 0.0
        self._score_n   = 0

        # ----- 주기 타이머 -----
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        # ----- 스쿼트 판정 파라미터 -----
        self._min_knee_in_phase = None
        self.score_overlay = ScoreOverlay(self)
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

        self.DOWN_TH = 120.0   # 내려갔다고 보는 무릎각 기준
        self.UP_TH   = 165.0   # 올라왔다고 보는 무릎각 기준
        self.DEBOUNCE_N = 3    # 연속 프레임 요구(노이즈 방지)

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
            self.btn_start.clicked.connect(lambda: self.ctx.cam.start())
            self.btn_stop.clicked.connect(lambda: self.ctx.cam.stop())
            self._connected = True

        self._session_started_ts = time.time()
        self._score_sum = 0.0
        self._score_n   = 0
        self.state = "UP"; self.reps = 0; self._down_frame = 0; self._up_frame = 0

        self._apply_processor()
        self.ctx.cam.start()

        if self.timer.isActive():
            self.timer.stop()
        self.timer.start(33)  

    def on_leave(self, ctx):
        if self.timer.isActive():
            self.timer.stop()
        ctx.cam.stop()

    def _toggle_skeleton(self, _):
        self._apply_processor()

    def _apply_processor(self):
        self.ctx.cam.set_process(self.proc if self.chk_skel.isChecked() else None)

    @staticmethod
    def _knees_from_meta(m):
        if not m:
            return None
        kL, kR = m.get("knee_l_deg"), m.get("knee_r_deg")
        if kL is None or kR is None:
            return None
        return float(kL), float(kR)

    def _knee_color_by_angle(self, ang: float) -> QColor:
        if ang <= 80:   return QColor(0, 128, 255)   # 파란색
        if ang <= 85:   return QColor(0, 200, 0)     # 녹색
        if ang <= 90:   return QColor(255, 255, 0)   # 노란색
        if ang <= 95:   return QColor(255, 140, 0)   # 주황
        return QColor(255, 0, 0)                     # 빨강

    def _score_by_angle(self, ang: float) -> int:  # 75~100 -> 100~0
        a0, s0 = 75.0, 100.0
        a1, s1 = 110.0, 0.0
        t = (ang - a0) / (a1 - a0)
        t = max(0.0, min(1.0, t))
        return int(round((1.0 - t) * s0 + t * s1))

    # ---------------- 상태 판정 ----------------
    def _is_down(self, knees):
        if knees is None:
            return False
        kL, kR = knees
        return (kL < self.DOWN_TH and kR < self.DOWN_TH)

    def _is_up(self, knees):
        if knees is None:
            return False
        kL, kR = knees
        return (kL >= self.UP_TH and kR >= self.UP_TH)

    # ---------------- 메인 루프 ----------------
    def _tick(self):
        self._update_video_and_info()

        m = self.ctx.cam.meta() or {}
        knees = self._knees_from_meta(m)

        is_down_now = self._is_down(knees)
        is_up_now   = self._is_up(knees)

        if is_down_now:
            self._down_frame += 1
            self._up_frame = 0
        elif is_up_now:
            self._up_frame += 1
            self._down_frame = 0
        else:
            self._down_frame = 0
            self._up_frame = 0

        if self.state == "DOWN" and knees is not None:
            cur_min = min(knees)
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

                ang_for_color = self._min_knee_in_phase if self._min_knee_in_phase is not None else (min(knees) if knees else 180.0)
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
             f"camera: {fmt_pair(knees)}")
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
            self.ctx.cam.stop()
        except Exception:
            pass

        if hasattr(self.ctx, "goto_summary"):
            self.ctx.goto_summary(summary)

    def _update_video_and_info(self):
        frame = self.ctx.cam.frame()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        draw_count_top_left(qimg, self.reps, "SQUAT")
        self.lbl_camera.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.lbl_camera.width(),
                self.lbl_camera.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        meta = self.ctx.cam.meta() or {}
        def fmt(x, f): return "-" if x is None else f.format(x)
        txt = ("Knee L/R: "
               f"{fmt(meta.get('knee_l_deg'), '{:5.1f}')}° / {fmt(meta.get('knee_r_deg'), '{:5.1f}')}°")
        self.info.setText(txt)
