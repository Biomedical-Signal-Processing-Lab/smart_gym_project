# views/select_page.py
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Qt
from core.page_base import PageBase

class SelectPage(PageBase):
    def __init__(self):
        super().__init__()

        self.btn_squat  = QPushButton("스쿼트")
        self.btn_plank  = QPushButton("플랭크")
        self.btn_lunge  = QPushButton("런지")
        self.btn_pushup = QPushButton("팔굽혀펴기")
        self.btn_info   = QPushButton("내 정보")

        self._buttons = [
            self.btn_squat,
            self.btn_plank,
            self.btn_lunge,
            self.btn_pushup,
            self.btn_info,
        ]

        for b in self._buttons:
            b.setFixedHeight(60)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn_squat.clicked.connect(self._go_squat)
        self.btn_plank.setEnabled(False)
        self.btn_lunge.setEnabled(False)
        self.btn_pushup.setEnabled(False)
        self.btn_info.clicked.connect(self._go_info)

        root = QVBoxLayout(self)
        root.addStretch(1)
        for b in self._buttons:
            root.addWidget(b, alignment=Qt.AlignHCenter)  
            root.addSpacing(20)
        root.addStretch(1)

        self._update_button_widths()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_button_widths()

    def _update_button_widths(self):
        target = int(self.width() * 0.5)
        for b in self._buttons:
            b.setFixedWidth(target)

    def _go_squat(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate("squat")

    def _go_info(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate("info")
