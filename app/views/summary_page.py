# views/summary_page.py (FINAL VERSION with background, 3 metric cards, integer score)
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QFrame, QSizePolicy
)
from datetime import timedelta
from core.page_base import PageBase
import os

def pretty_hms(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def score_color(score: int) -> str:
    try:
        s = int(score)
    except Exception:
        return "#777"
    if s >= 90: return "#16a34a"
    if s >= 80: return "#2563eb"
    if s >= 70: return "#ea580c"
    return "#6b7280"

def asset_path(*parts) -> str:
    """Utility to resolve asset path"""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, *parts)

# -------------------- MetricCard --------------------
class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "-", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("MetricTitle")
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("MetricValue")

        row = QVBoxLayout(self)
        row.setContentsMargins(18, 16, 18, 16)
        row.setSpacing(4)
        row.addWidget(self.title_lbl)
        row.addWidget(self.value_lbl)
        row.addStretch(1)

    def setValue(self, text: str):
        self.value_lbl.setText(text)

class ExerciseCard(QFrame):
    def __init__(self, name: str = "-", reps: int = None, score: float = None, placeholder=False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ExerciseCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.placeholder = placeholder

        self.name_lbl = QLabel(name)
        self.name_lbl.setObjectName("ExName")

        score_display = "-" if placeholder else f"{int(round(score or 0))}점"
        reps_display = "-" if placeholder else f"{int(reps or 0)}회"

        self.count_value = QLabel(reps_display)
        self.count_value.setObjectName("ExCount")
        self.score_value = QLabel(score_display)
        self.score_value.setObjectName("ExScore")

        if not placeholder:
            self.score_value.setStyleSheet(f"color: {score_color(int(round(score or 0)))};")
        else:
            self.setStyleSheet("QFrame#ExerciseCard { opacity: 0.4; }")

        top = QHBoxLayout()
        top.addWidget(self.name_lbl)
        top.addStretch(1)

        cnt_box = self._pill_block("횟수", self.count_value)
        scr_box = self._pill_block("점수", self.score_value)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(cnt_box)
        bottom.addWidget(scr_box)
        bottom.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        lay.addLayout(top)
        lay.addLayout(bottom)

    def _pill_block(self, title: str, value_label: QLabel) -> QWidget:
        box = QFrame()
        box.setObjectName("PillBox")
        row = QHBoxLayout(box)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        t = QLabel(title)
        t.setObjectName("PillTitle")
        row.addWidget(t)
        row.addWidget(value_label)
        return box

    def update_data(self, count: int, score: float):
        if not self.placeholder:  # ignore if it's a placeholder
            self.count_value.setText(f"{count}회")
            self.score_value.setText(f"{score}점")
            self.score_value.setStyleSheet(f"color: {score_color(score)};")

class SummaryPage(PageBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SummaryPage")
        self._summary = {}
        self.ctx = None

        self._bg_pix = None
        self.bg = QLabel(self)
        self.bg.setScaledContents(False)
        self.bg.lower()
        bg_path = asset_path("assets", "background", "bg_gym.jpg")
        if os.path.exists(bg_path):
            pm = QPixmap(bg_path)
            if not pm.isNull():
                self._bg_pix = pm
                self._rescale_bg()

        self.panel = QFrame(self)
        self.panel.setObjectName("glassPanel")
        self.panel.setAttribute(Qt.WA_StyledBackground, True)

        root = QVBoxLayout(self.panel)
        root.setContentsMargins(28, 28, 28, 28)
        root.setSpacing(18)

        top = QFrame()
        top.setObjectName("TopBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(22, 18, 22, 18)
        top_lay.setSpacing(14)

        self.title = QLabel("오늘의 운동 완료!")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        top_lay.addWidget(self.title, 1)

        self.btn_profile = QPushButton("내 정보")
        self.btn_profile.setObjectName("BtnUser")
        self.btn_profile.setFixedHeight(60)
        self.btn_profile.clicked.connect(self._on_profile)

        self.btn_retry = QPushButton("다시하기")
        self.btn_retry.setObjectName("BtnStart")
        self.btn_retry.setFixedHeight(60)
        self.btn_retry.clicked.connect(self._on_retry)

        top_lay.addWidget(self.btn_profile)
        top_lay.addWidget(self.btn_retry)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)

        self.total_time_card = MetricCard("총 운동 시간", "00:00")
        self.total_reps_card = MetricCard("총 운동 횟수", "0회")
        self.avg_score_card = MetricCard("평균 점수", "0점")  

        for w in (self.total_time_card, self.total_reps_card, self.avg_score_card):
            w.setMinimumHeight(90)
            metrics_row.addWidget(w)

        section = QLabel("운동별 상세 결과")
        section.setObjectName("SectionTitle")

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(12)
        self.exercise_cards: list[ExerciseCard] = []

        root.addWidget(top)
        root.addLayout(metrics_row)
        root.addWidget(section)
        root.addLayout(self.grid)

        self.setStyleSheet(self._stylesheet())

    def on_enter(self, ctx):
        self.ctx = ctx

    def set_data(self, summary: dict):
        self._summary = dict(summary or {})
        self._render_with_summary()

    def _render_with_summary(self):
        if not self._summary:
            return

        d = self._summary
        total_seconds = d.get("duration_sec", 0)
        per_list = d.get("exercises") or []

        total_reps = sum(int(x.get("reps", 0)) for x in per_list)
        avg_score = 0
        if total_reps > 0:
            w_sum = sum(int(x.get("reps", 0)) * float(x.get("avg", x.get("avg_score", 0.0))) for x in per_list)
            avg_score = w_sum / total_reps

        self.total_time_card.setValue(pretty_hms(total_seconds))
        self.total_reps_card.setValue(f"{total_reps}회")
        self.avg_score_card.setValue(f"{int(round(avg_score))}점") 

        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.exercise_cards.clear()

        limited = per_list[:8]
        while len(limited) < 8:
            limited.append(None)

        positions = [(r, c) for r in range(4) for c in range(2)]
        for (r, c), data in zip(positions, limited):
            if data:
                name = data.get("name", "-")
                reps = int(data.get("reps", 0))
                score = float(data.get("avg", data.get("avg_score", 0.0)))
                card = ExerciseCard(name=name, reps=reps, score=score, placeholder=False)
            else:
                card = ExerciseCard(name="-", reps=0, score=0, placeholder=True)
            self.exercise_cards.append(card)
            self.grid.addWidget(card, r, c)

    # -------------------- Events --------------------
    def _on_retry(self):
        ex = (self._summary or {}).get("exercise")
        if self.ctx and hasattr(self.ctx, "restart_current_exercise"):
            self.ctx.restart_current_exercise(ex)

    def _on_profile(self):
        if self.ctx and hasattr(self.ctx, "goto_profile"):
            self.ctx.goto_profile()

    # -------------------- Resize & Layout --------------------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_bg()
        self._layout_panel()

    def _layout_panel(self):
        w, h = self.width(), self.height()
        target_w = max(int(w * 0.95), 1100)
        target_h = max(int(h * 0.95), 720)
        x = (w - target_w) // 2
        y = (h - target_h) // 2
        self.panel.setGeometry(x, y, target_w, target_h)

    def _rescale_bg(self):
        if self._bg_pix:
            self.bg.setGeometry(self.rect())
            scaled = self._bg_pix.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            self.bg.setPixmap(scaled)

    # -------------------- Stylesheet --------------------
    def _stylesheet(self) -> str:
        return """
        #glassPanel {
            background: rgba(255,255,255,1);
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.25);
        }
        #TopBar {
            background: rgba(138, 43, 226, 0.8);
            border-radius: 20px;
        }
        #Title {
            color: white;
            font-size: 44px;
            font-weight: 500;
            letter-spacing: 1px;
        }
        #BtnUser {
            background: rgba(40, 167, 69, 1);
            color: white;
            border: none;
            padding: 0 22px;
            border-radius: 14px;
            font-size: 24px;
            font-weight: 600;
        }
        #BtnUser:hover { background: rgba(50, 200, 85, 1); }

        #BtnStart {
            background: rgba(0, 123, 255, 1);
            color: white;
            border: none;
            padding: 0 24px;
            border-radius: 14px;
            font-size: 24px;
            font-weight: 700;
        }
        #BtnStart:hover { background: rgba(0, 105, 230, 1); }

        #MetricCard {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(52, 142, 219, 0.25), stop:1 rgba(52, 142, 219, 0.4));
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
        }
        #MetricTitle {
            color: #0b3b71;
            font-size: 30px;
            font-weight: 500;
        }
        #MetricValue {
            color: #0f172a;
            font-size: 32px;
            font-weight: 900;
        }
        #SectionTitle {
            color: rgba(0, 123, 255, 1);
            font-size: 60px;
            font-weight: 500;
            padding-left: 6px;
        }
        #ExerciseCard {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 18px;
        }
        #ExName { font-size: 40px; font-weight: 500; color: #111827; }
        #ExCount { font-size: 25px; font-weight: 500; color: #0f172a; }
        #ExScore { font-size: 25px; font-weight: 500; }

        #PillBox {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #eef7ff, stop:1 #e6f2ff);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 14px;
        }
        #PillTitle { color: #475569; font-size: 25px; font-weight: 500; }
        """
