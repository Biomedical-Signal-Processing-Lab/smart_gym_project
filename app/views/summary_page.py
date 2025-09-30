# views/summary_page.py
import time
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QGroupBox
from PySide6.QtCore import Qt
from core.page_base import PageBase

def _fmt_duration(sec: int) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}시간 {m}분 {s}초"
    if m: return f"{m}분 {s}초"
    return f"{s}초"

class SummaryPage(PageBase):
    def __init__(self):
        super().__init__()
        self.data = {}

        self.title = QLabel("운동 요약")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 28px; font-weight: 800;")

        self.lbl_main = QLabel("-")
        self.lbl_main.setAlignment(Qt.AlignCenter)
        self.lbl_main.setStyleSheet("font-size: 18px;")

        self.btn_back = QPushButton("메인으로")
        self.btn_restart = QPushButton("다시 시작")

        b = QHBoxLayout()
        b.addStretch(1); b.addWidget(self.btn_restart); b.addWidget(self.btn_back); b.addStretch(1)

        root = QVBoxLayout(self)
        root.addWidget(self.title)
        grp = QGroupBox("결과")
        g = QVBoxLayout(); g.addWidget(self.lbl_main); grp.setLayout(g)
        root.addWidget(grp)
        root.addLayout(b)

        self.btn_back.clicked.connect(lambda: hasattr(self, "ctx") and getattr(self.ctx, "goto_main", lambda: None)())
        self.btn_restart.clicked.connect(lambda: hasattr(self, "ctx") and getattr(self.ctx, "restart_current_exercise", lambda: None)(self.data.get("exercise")))

    def on_enter(self, ctx):
        self.ctx = ctx
        self._render()

    def set_data(self, summary: dict):
        self.data = dict(summary or {})
        # 이미 표시 중이면 즉시 반영
        if self.isVisible():
            self._render()

    def _render(self):
        d = self.data or {}
        ex = d.get("exercise", "-")
        reps = d.get("reps", 0)
        avg = d.get("avg_score", 0.0)
        dur = _fmt_duration(d.get("duration_sec", 0))
        started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.get("started_at", time.time())))
        ended   = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d.get("ended_at",   time.time())))

        self.title.setText(f"{ex.upper()} 요약")
        self.lbl_main.setText(
            f"운동: {ex}\n"
            f"총 횟수: {reps}\n"
            f"평균 점수: {avg}\n"
            f"운동 시간: {dur}\n"
            f"시작: {started}\n"
            f"종료: {ended}"
        )
