# views/summary_page.py
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QScrollArea, QFrame, QSizePolicy, QSpacerItem, QGraphicsDropShadowEffect
)
from datetime import timedelta
from core.page_base import PageBase

def pretty_hms(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    return str(timedelta(seconds=int(seconds)))

class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet("""
            QFrame#Card {
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid rgba(0,0,0,0.06);
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(Qt.GlobalColor.lightGray)
        self.setGraphicsEffect(shadow)

class MetricCard(Card):
    def __init__(self, title: str, value: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(6)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet("color:#667085;")
        self.title_lbl.setFont(QFont("Inter", 12, QFont.Weight.Medium))

        self.value_lbl = QLabel(value)
        self.value_lbl.setStyleSheet("color:#111827;")
        f = QFont("Inter", 28, QFont.Weight.Bold)
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 103)
        self.value_lbl.setFont(f)

        self.subtitle_lbl = QLabel(subtitle)
        self.subtitle_lbl.setStyleSheet("color:#98a2b3;")
        self.subtitle_lbl.setFont(QFont("Inter", 11))

        lay.addWidget(self.title_lbl)
        lay.addWidget(self.value_lbl)
        if subtitle:
            lay.addWidget(self.subtitle_lbl)
        lay.addStretch(1)

    def setValue(self, value: str):
        self.value_lbl.setText(value)

class ExerciseCard(Card):
    def __init__(self, name: str, reps: int, avg_score: float, seconds: int, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(12)

        name_box = QVBoxLayout()
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("color:#1f2937;")
        name_lbl.setFont(QFont("Inter", 18, QFont.Weight.DemiBold))
        name_box.addWidget(name_lbl)
        name_box.addStretch(1)

        grid = QGridLayout()
        grid.setVerticalSpacing(4)
        grid.setHorizontalSpacing(16)

        def mk_pair(label, value):
            l = QLabel(label)
            l.setStyleSheet("color:#667085;")
            l.setFont(QFont("Inter", 11))
            v = QLabel(value)
            v.setStyleSheet("color:#1f2937;")
            v.setFont(QFont("Inter", 16, QFont.Weight.Bold))
            return l, v

        l1, v1 = mk_pair("횟수", f"{reps} 회")
        l2, v2 = mk_pair("평균 점수", f"{avg_score:.1f} 점")
        l3, v3 = mk_pair("운동 시간", pretty_hms(seconds))

        grid.addWidget(l1, 0, 0)
        grid.addWidget(v1, 1, 0)
        grid.addWidget(l2, 0, 1)
        grid.addWidget(v2, 1, 1)
        grid.addWidget(l3, 0, 2)
        grid.addWidget(v3, 1, 2)

        lay.addLayout(name_box, 1)
        lay.addLayout(grid, 2)

class SummaryPage(PageBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SummaryPage")
        self._summary = {}
        self.ctx = None

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#f5f7fb"))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        self.setStyleSheet(self._root_qss())

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        header = QLabel("오늘의 운동 결과")
        header.setObjectName("Header")
        header.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(header)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        self.total_time_card = MetricCard("⏱ 총 운동시간", pretty_hms(0))
        self.avg_score_card   = MetricCard("⭐ 평균 점수", f"{0.0:.1f} 점")
        self.total_time_card.setMinimumHeight(110)
        self.avg_score_card.setMinimumHeight(110)
        top_row.addWidget(self.total_time_card, 1)
        top_row.addWidget(self.avg_score_card, 1)
        root.addLayout(top_row)

        mid_card = Card()
        mid_lay = QVBoxLayout(mid_card)
        mid_lay.setContentsMargins(18, 18, 18, 18)
        mid_lay.setSpacing(12)

        title_row = QHBoxLayout()
        title_lbl = QLabel("운동별 요약")
        title_lbl.setStyleSheet("color:#111827;")
        title_lbl.setFont(QFont("Inter", 16, QFont.Weight.DemiBold))
        title_row.addWidget(title_lbl)
        title_row.addStretch(1)

        self.up_btn = self._nav_button("▲ 위로")
        self.down_btn = self._nav_button("▼ 아래로")
        self.up_btn.clicked.connect(lambda: self._scroll_by(-1))
        self.down_btn.clicked.connect(lambda: self._scroll_by(+1))
        title_row.addWidget(self.up_btn)
        title_row.addWidget(self.down_btn)
        mid_lay.addLayout(title_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: 0; }
            QScrollArea > QWidget { background: transparent; }
            QScrollArea QWidget { background: transparent; }
        """)

        self.list_container = QWidget()
        self.list_container.setStyleSheet("background: transparent;")
        self.list_lay = QVBoxLayout(self.list_container)
        self.list_lay.setContentsMargins(6, 6, 6, 6)
        self.list_lay.setSpacing(10)
        self.scroll.setWidget(self.list_container)
        mid_lay.addWidget(self.scroll)
        root.addWidget(mid_card, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        self.retry_btn   = self._cta_button("다시하기")
        self.home_btn    = self._cta_button("메인으로")
        self.profile_btn = self._cta_button("내정보")
        bottom.addWidget(self.retry_btn, 1)
        bottom.addWidget(self.home_btn, 1)
        bottom.addWidget(self.profile_btn, 1)
        root.addLayout(bottom)

        self.btn_restart = self.retry_btn
        self.btn_back    = self.home_btn

        self.home_btn.clicked.connect(self._on_home)
        self.retry_btn.clicked.connect(self._on_retry)
        self.profile_btn.clicked.connect(self._on_profile)

    # Router hook
    def on_enter(self, ctx):
        self.ctx = ctx

    def set_data(self, summary: dict):
        self._summary = dict(summary or {})
        self._render_with_summary()

    def _on_home(self):
        if self.ctx and hasattr(self.ctx, "goto_main"):
            self.ctx.goto_main()

    def _on_retry(self):
        ex = (self._summary or {}).get("exercise")
        if self.ctx and hasattr(self.ctx, "restart_current_exercise"):
            self.ctx.restart_current_exercise(ex)

    def _on_profile(self):
        if self.ctx and hasattr(self.ctx, "goto_profile"):
            self.ctx.goto_profile()

    def _render_with_summary(self):
        if not self._summary:
            return
        d = self._summary
        total_seconds = d.get("duration_sec", 0)
        avg_score = d.get("avg_score", 0.0)
        per_list = d.get("per_exercises") or []

        if per_list:
            total_seconds = sum(x.get("sec", 0) for x in per_list)
            w_sum = sum(x.get("avg", 0.0) * x.get("reps", 0) for x in per_list)
            reps = sum(x.get("reps", 0) for x in per_list) or 1
            avg_score = (w_sum / reps) if reps else 0.0

        self.total_time_card.setValue(pretty_hms(total_seconds))
        self.avg_score_card.setValue(f"{avg_score:.1f} 점")

        while self.list_lay.count():
            it = self.list_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        rows = per_list or [{
            "name": d.get("exercise", "운동"),
            "reps": d.get("reps", 0),
            "avg": d.get("avg_score", 0.0),
            "sec": d.get("duration_sec", 0),
        }]

        for item in rows:
            card = ExerciseCard(
                f"🏋️ {item.get('name','운동')}",
                int(item.get("reps", 0)),
                float(item.get("avg", item.get("avg_score", 0.0))),
                int(item.get("sec", item.get("duration_sec", 0)))
            )
            card.setMinimumHeight(96)
            self.list_lay.addWidget(card)

        self.list_lay.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

    def _root_qss(self) -> str:
        return """
        QWidget#SummaryPage {
            background: #f5f7fb;
            font-family: Inter, Pretendard, Apple SD Gothic Neo, "Noto Sans KR", Malgun Gothic, sans-serif;
            color: #1f2937;
        }
        QLabel { background: transparent; }
        QLabel#Header {
            font-size: 28px; font-weight: 800; color: #111827;
            letter-spacing: 0.5px;
        }
        QScrollBar:vertical {
            background: #e7ecf5;
            width: 12px; margin: 6px; border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #c8d1e1;
            min-height: 30px; border-radius: 6px;
        }
        QPushButton[cssClass="Nav"] {
            background: #eef2f7;
            border: 1px solid #d7dee9;
            border-radius: 12px; padding: 10px 14px; color: #334155;
            font-size: 14px; font-weight: 600;
        }
        QPushButton[cssClass="Nav"]:hover { background: #e6ebf3; }
        QPushButton[cssClass="Nav"]:pressed { background: #dde4ef; }

        QPushButton[cssClass="CTA"] {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #7cc6ff, stop:1 #98a8ff);
            border: none; border-radius: 16px;
            padding: 18px 24px; font-size: 18px; font-weight: 800; color: white;
        }
        QPushButton#Alt[cssClass="CTA"] { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #34d399, stop:1 #10b981); }
        QPushButton#Danger[cssClass="CTA"] { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fb7185, stop:1 #f97316); }
        QPushButton[cssClass="CTA"]:pressed { filter: brightness(0.95); }
        """

    def _nav_button(self, text: str) -> QPushButton:
        b = QPushButton(text)
        b.setProperty("cssClass", "Nav")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setMinimumHeight(44)
        b.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setMinimumWidth(88)
        return b

    def _cta_button(self, text: str) -> QPushButton:
        b = QPushButton(text)
        b.setProperty("cssClass", "CTA")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setMinimumHeight(64)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return b

    def _scroll_by(self, pages: int):
        bar = self.scroll.verticalScrollBar()
        step = self.scroll.viewport().height() * 0.85
        bar.setValue(int(bar.value() + pages * step))
