# views/start_page.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSizePolicy
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from core.page_base import PageBase

class StartPage(PageBase):
    def __init__(self):
        super().__init__()
        title = QLabel("최고의 스쿼트"); title.setAlignment(Qt.AlignCenter)
        f = QFont(); f.setPointSize(36); f.setBold(True); title.setFont(f)
        subtitle = QLabel("당신의 운동자세를 분석합니다"); subtitle.setAlignment(Qt.AlignCenter)

        btn = QPushButton("시작하기"); btn.setFixedHeight(48); btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.clicked.connect(self._go_video)

        v = QVBoxLayout(self)
        v.addStretch(2); v.addWidget(title); v.addWidget(subtitle); v.addStretch(1)
        h = QHBoxLayout(); h.addStretch(1); h.addWidget(btn); h.addStretch(1); v.addLayout(h); v.addStretch(3)

    def _go_video(self):
        # 부모의 Router를 찾아서 이동
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate("video")
