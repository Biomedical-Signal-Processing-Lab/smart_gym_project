# views/select_page.py
from PySide6.QtWidgets import QVBoxLayout, QWidget, QPushButton, QSizePolicy
from PySide6.QtCore import Qt
from core.page_base import PageBase
from ui.topbar import TopBar

class SelectPage(PageBase):
    def __init__(self):
        super().__init__()

        self.topbar = TopBar(self, show_back=False)

        self.btn_squat  = QPushButton("스쿼트")
        self.btn_plank  = QPushButton("플랭크")
        self.btn_lunge  = QPushButton("런지")
        self.btn_pushup = QPushButton("팔굽혀펴기")

        self._buttons = [self.btn_squat, self.btn_plank, self.btn_lunge, self.btn_pushup]

        for b in self._buttons:
            b.setFixedHeight(60)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn_squat.clicked.connect(lambda: self._goto("squat"))
        self.btn_plank.setEnabled(False)
        self.btn_lunge.setEnabled(False)
        self.btn_pushup.setEnabled(False)

        content = QWidget(self)
        cv = QVBoxLayout(content)
        cv.setContentsMargins(0, 24, 0, 24)
        cv.addStretch(1)
        for b in self._buttons:
            cv.addWidget(b, alignment=Qt.AlignHCenter)
            cv.addSpacing(20)
        cv.addStretch(1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.topbar)
        root.addWidget(content, 1)

        self._update_button_widths()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_button_widths()

    def _update_button_widths(self):
        target = int(self.width() * 0.5)
        for b in self._buttons:
            b.setFixedWidth(target)

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)
