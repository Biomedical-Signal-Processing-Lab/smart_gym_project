# views/squat_page.py
import time, cv2
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QVBoxLayout
from core.page_base import PageBase
from core.pose_manager import PoseProcessor
from ui.overlay_painter import VideoCanvas, CanvasHUD
from ui.score_painter import ScoreOverlay

APP_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_DIR / "models" / "yolov8m_pose.onnx"

DEMO_EXERCISES = [
    {"name": "스쿼트",   "reps": 32, "avg": 93.2, "sec": 1180},
    {"name": "런지",     "reps": 24, "avg": 88.5, "sec": 860},
    {"name": "벤치프레스","reps": 18, "avg": 91.0, "sec": 740},
    {"name": "데드리프트","reps": 20, "avg": 89.1, "sec": 900},
    {"name": "풀업",     "reps": 12, "avg": 85.7, "sec": 520},
    {"name": "푸시업",   "reps": 40, "avg": 94.0, "sec": 1100},
    {"name": "사이드 레이즈","reps": 30, "avg": 90.2, "sec": 780},
    {"name": "플랭크",   "reps": 4,  "avg": 92.0, "sec": 480},
]

class ExercisePage(PageBase):
    DOWN_TH    = 120.0
    UP_TH      = 165.0
    DEBOUNCE_N = 3

    def __init__(self):
        super().__init__()
        self.setObjectName("ExercisePage")

        self.state = "UP"
        self.reps = 0
        self._down_frame = 0
        self._up_frame = 0
        self._min_knee_in_phase = None
        self._score_sum = 0.0
        self._score_n   = 0
        self._session_started_ts = None

        self.proc = PoseProcessor(
            onnx_path=str(MODEL_PATH),
            conf_thres=0.10,
            draw_landmarks=True,
            name="pose_squat",
        )

        self.canvas = VideoCanvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        self.canvas.set_fit_mode("cover")

        self.hud = CanvasHUD(self.canvas, count_label_text="운동종류")
        self.hud.endClicked.connect(self._end_clicked)

        self.score_overlay = ScoreOverlay(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.canvas, 1)

        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.PAGE_FPS_MS = 33  # ~30fps

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    def _build_summary(self):
        per_list = DEMO_EXERCISES
        total_seconds = sum(x["sec"] for x in per_list)
        w_sum = sum(x["avg"] * x["reps"] for x in per_list)
        reps_sum = sum(x["reps"] for x in per_list) or 1
        avg_total = w_sum / reps_sum

        ended_at = time.time()
        return {
            "duration_sec": int(total_seconds),
            "avg_score": round(avg_total, 1),
            "per_exercises": per_list,
            "exercise": "squat",
            "reps": self.reps,
            "started_at": self._session_started_ts,
            "ended_at": ended_at,
        }

    def _end_clicked(self):
        if self.timer.isActive():
            self.timer.stop()
        try:
            self.ctx.cam.stop()
        except Exception:
            pass

        summary = self._build_summary()
        try:
            if hasattr(self.ctx, "save_workout_session"):
                self.ctx.save_workout_session(summary)
        except Exception as e:
            print(f"[WARN] workout save failed: {e}")

        if hasattr(self.ctx, "goto_summary"):
            self.ctx.goto_summary(summary)

        self.canvas.clear_overlays()
        self.hud.teardown()

    def on_enter(self, ctx):
        self.ctx = ctx
        self._session_started_ts = time.time()
        self._score_sum = 0.0
        self._score_n = 0
        self._reset_state()

        self.ctx.cam.set_process(self.proc)
        self.ctx.cam.start()

        if self.timer.isActive():
            self.timer.stop()
        self.timer.start(self.PAGE_FPS_MS)
        self.hud.mount()

    def on_leave(self, ctx):
        if self.timer.isActive():
            self.timer.stop()
        try:
            ctx.cam.stop()
        except Exception:
            pass
        self.canvas.clear_overlays()
        self.hud.teardown()

    def _tick(self):
        frame = self.ctx.cam.frame()
        meta = self.ctx.cam.meta() or {}
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.canvas.set_frame(qimg)

        self._update_from_meta(meta)

    def _reset_state(self):
        self.state = "UP"
        self.reps = 0
        self._down_frame = 0
        self._up_frame = 0
        self._min_knee_in_phase = None
        self.hud.set_count(0)

    def _knees_from_meta(self, m):
        kL, kR = m.get("knee_l_deg"), m.get("knee_r_deg")
        if kL is None or kR is None: 
            return None
        return float(kL), float(kR)

    def _knee_color_by_angle(self, ang: float) -> QColor:
        if ang <= 80:   return QColor(0, 128, 255)
        if ang <= 85:   return QColor(0, 200, 0)
        if ang <= 90:   return QColor(255, 255, 0)
        if ang <= 95:   return QColor(255, 140, 0)
        return QColor(255, 0, 0)

    def _score_by_angle(self, ang: float) -> int:
        a0, s0 = 75.0, 100.0
        a1, s1 = 110.0, 0.0
        t = max(0.0, min(1.0, (ang - a0) / (a1 - a0)))
        return round((1.0 - t) * s0 + t * s1)

    def _update_from_meta(self, meta: dict):
        knees = self._knees_from_meta(meta)
        is_down_now = knees and (knees[0] < self.DOWN_TH and knees[1] < self.DOWN_TH)
        is_up_now   = knees and (knees[0] >= self.UP_TH and knees[1] >= self.UP_TH)

        if is_down_now:
            self._down_frame += 1; self._up_frame = 0
        elif is_up_now:
            self._up_frame += 1; self._down_frame = 0
        else:
            self._down_frame = 0; self._up_frame = 0

        if self.state == "DOWN" and knees is not None:
            cur_min = min(knees)
            self._min_knee_in_phase = cur_min if self._min_knee_in_phase is None else min(self._min_knee_in_phase, cur_min)

        if self.state == "UP":
            if self._down_frame >= self.DEBOUNCE_N:
                self.state = "DOWN"
                self._min_knee_in_phase = None
        else:
            if self._up_frame >= self.DEBOUNCE_N:
                self.state = "UP"
                self.reps += 1
                self.hud.set_count(self.reps)

                ang = self._min_knee_in_phase if self._min_knee_in_phase is not None else (min(knees) if knees else 180.0)
                color = self._knee_color_by_angle(ang)
                score = self._score_by_angle(ang)
                self._score_sum += float(score); self._score_n += 1
                self.score_overlay.show_score(str(score), 100, text_qcolor=color)
                self._min_knee_in_phase = None

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()
