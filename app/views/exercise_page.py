# views/squat_page.py
import time, cv2
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QVBoxLayout
from core.page_base import PageBase
from core.pose_manager import PoseProcessor
from ui.overlay_painter import VideoCanvas, ExerciseCard, ScoreAdvicePanel, ActionButtons
from ui.score_painter import ScoreOverlay

APP_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_DIR / "models" / "yolov8m_pose.onnx"

DEMO_EXERCISES = [
    {"name": "스쿼트",   "reps": 32, "avg": 93.2},
    {"name": "런지",     "reps": 24, "avg": 88.5},
    {"name": "벤치프레스","reps": 18, "avg": 91.0},
    {"name": "데드리프트","reps": 20, "avg": 89.1},
    {"name": "풀업",     "reps": 12, "avg": 85.7},
    {"name": "푸시업",   "reps": 40, "avg": 94.0},
    {"name": "사이드 레이즈", "reps": 30, "avg": 90.2},
    {"name": "숨쉬기", "reps": 35, "avg": 81.2},
    {"name": "걷기", "reps": 45, "avg": 73.4},
    {"name": "뛰기", "reps": 65, "avg": 66.2}
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

        self.card = ExerciseCard("휴식중")
        self.panel = ScoreAdvicePanel()
        self.panel.set_avg(0)
        self.panel.set_advice("올바른 자세로 준비하세요.")
        self.actions = ActionButtons()
        self.actions.endClicked.connect(self._end_clicked)
        self.actions.infoClicked.connect(self._info_clicked)

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

    def _mount_overlays(self):
        self.canvas.clear_overlays()
        self.canvas.add_overlay(self.card, anchor="top-left")
        self.canvas.add_overlay(self.panel, anchor="top-right")
        self.canvas.add_overlay(self.actions, anchor="bottom-right")
        self.card.show()
        self.panel.show()
        self.actions.show()
        self._sync_panel_sizes()

    def _sync_panel_sizes(self):
        W, H = self.width(), self.height()
        target_w = int(max(320, min(W * 0.26, 460)))
        target_h = int(target_w * 0.90)
        self.card.setFixedSize(target_w, target_h)
        self.panel.setFixedSize(target_w, target_h)

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    def _build_summary(self):
        per_list = DEMO_EXERCISES

        w_sum = sum(float(x.get("avg", 0.0)) * int(x.get("reps", 0)) for x in per_list)
        reps_sum = sum(int(x.get("reps", 0)) for x in per_list) or 1
        avg_total = w_sum / reps_sum

        ended_at = time.time()
        started_at = self._session_started_ts or ended_at
        duration_sec = int(max(0, ended_at - started_at))

        return {
            "duration_sec": duration_sec,          
            "avg_score": round(avg_total, 1),      
            "exercises": per_list 
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

    def _info_clicked(self):
        try:
            if hasattr(self.ctx, "goto_profile"):
                self.ctx.goto_profile()
        except Exception:
            pass

    def on_enter(self, ctx):
        self.ctx = ctx
        self._session_started_ts = time.time()
        self._score_sum = 0.0
        self._score_n = 0
        self._reset_state()

        title_text = getattr(self.ctx, "current_exercise", None) or "휴식중"
        self.card.set_title(title_text)

        self._mount_overlays()

        self.ctx.cam.set_process(self.proc)
        self.ctx.cam.start()

        if self.timer.isActive():
            self.timer.stop()
        self.timer.start(self.PAGE_FPS_MS)

    def on_leave(self, ctx):
        if self.timer.isActive():
            self.timer.stop()
        try:
            ctx.cam.stop()
        except Exception:
            pass
        self.canvas.clear_overlays()

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
        self.card.set_count(0)
        self.panel.set_avg(0)
        self.panel.set_advice("올바른 자세로 준비하세요.")

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

    def _pick_advice(self, knees_min: float | None, meta: dict) -> str:
        if knees_min is None:
            return "무릎이 화면에 잘 보이도록 조금 더 뒤로 물러나세요."
        if knees_min < 80:
            return "너무 깊습니다. 허리가 말리지 않게 80~90° 구간을 노려보세요."
        if knees_min > 110:
            return "조금 더 내려가 보세요. 무릎 각도 90~100°가 좋아요."
        kl, kr = meta.get("knee_l_deg"), meta.get("knee_r_deg")
        if isinstance(kl, (int, float)) and isinstance(kr, (int, float)) and abs(kl - kr) > 8:
            return "좌우 무릎 각도 차이가 커요. 체중을 중앙에 실어보세요."
        return "좋아요! 같은 리듬으로 1초에 1회 정도 유지해보세요."

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
                self.card.set_count(self.reps)

                ang = self._min_knee_in_phase if self._min_knee_in_phase is not None else (min(knees) if knees else 180.0)
                color = self._knee_color_by_angle(ang)
                score = self._score_by_angle(ang)
                self._score_sum += float(score); self._score_n += 1

                avg = (self._score_sum / self._score_n) if self._score_n else 0.0
                self.panel.set_avg(avg)
                self.panel.set_advice(self._pick_advice(self._min_knee_in_phase, meta))

                self.score_overlay.show_score(str(score), 100, text_qcolor=color)
                self._min_knee_in_phase = None

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_panel_sizes()
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()
