# views/squat_page.py
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QPushButton, QLabel
from ui.overlay_painter import VideoCanvas
from views.exercise_base import ExerciseBasePage
from ui.score_painter import ScoreOverlay

class SquatPage(ExerciseBasePage):
    DOWN_TH = 120.0
    UP_TH   = 165.0
    DEBOUNCE_N = 3

    def __init__(self):
        super().__init__(title="squat")
        self.state = "UP"
        self.reps = 0
        self._down_frame = 0
        self._up_frame = 0
        self._min_knee_in_phase = None

        self.score_overlay = ScoreOverlay(self)
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

        self._btn_end_top = QPushButton("운동 종료", self)
        self._btn_end_top.clicked.connect(self._end_workout)
        self._btn_end_top.setStyleSheet(
            "background: rgba(0,0,0,120); color: white; border-radius: 10px; padding: 6px 12px;"
        )

        self.lbl_count_overlay = QLabel(self)
        self.lbl_count_overlay.setText("SQUAT: 0")
        self.lbl_count_overlay.setStyleSheet("""
            QLabel {
                background: rgba(0,0,0,120);
                color: white;
                padding: 6px 10px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 18px;
            }
        """)

    def build_overlays(self, canvas: VideoCanvas):
        canvas.add_overlay(self._btn_end_top, anchor="top-right")
        canvas.add_overlay(self.lbl_count_overlay, anchor="top-left")
        self.score_overlay.setGeometry(self.rect())
        self.score_overlay.raise_()

    def _hook_reset(self):
        self.state = "UP"
        self.reps = 0
        self._down_frame = 0
        self._up_frame = 0
        self._min_knee_in_phase = None
        if hasattr(self, "lbl_count_overlay"):
            self.lbl_count_overlay.setText("SQUAT: 0")

    def _knees_from_meta(self, m):
        kL, kR = m.get("knee_l_deg"), m.get("knee_r_deg")
        if kL is None or kR is None: return None
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
        t = max(0.0, min(1.0, (ang - a0)/(a1 - a0)))
        return round((1.0 - t) * s0 + t * s1)

    def on_meta(self, meta: dict):
        knees = self._knees_from_meta(meta)
        is_down_now = knees and (knees[0] < self.DOWN_TH and knees[1] < self.DOWN_TH)
        is_up_now   = knees and (knees[0] >= self.UP_TH and knees[1] >= self.UP_TH)

        if is_down_now:   self._down_frame += 1; self._up_frame = 0
        elif is_up_now:   self._up_frame   += 1; self._down_frame = 0
        else:             self._down_frame = 0;  self._up_frame   = 0

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

                if hasattr(self, "lbl_count_overlay"):
                    self.lbl_count_overlay.setText(f"SQUAT: {self.reps}")

                ang = self._min_knee_in_phase if self._min_knee_in_phase is not None else (min(knees) if knees else 180.0)
                color = self._knee_color_by_angle(ang)
                score = self._score_by_angle(ang)
                self._score_sum += float(score); self._score_n += 1
                self.score_overlay.show_score(str(score), 100, text_qcolor=color)
                self._min_knee_in_phase = None

    def format_bottom_text(self, meta: dict) -> str:
        def fmt(x): return "-" if x is None else f"{x:5.1f}°"
        return f"Knee L/R: {fmt(meta.get('knee_l_deg'))} / {fmt(meta.get('knee_r_deg'))}"

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "score_overlay") and self.score_overlay is not None:
            self.score_overlay.setGeometry(self.rect())
            self.score_overlay.raise_()
