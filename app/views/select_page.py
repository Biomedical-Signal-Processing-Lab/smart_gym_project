# app/views/select_page.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from core.page_base import PageBase

class SelectPage(PageBase):
    def __init__(self):
        super().__init__()

        self.btn_squat  = QPushButton("Squat")
        self.btn_plank  = QPushButton("Plank")
        self.btn_pushup = QPushButton("Push up")
        self.btn_soon   = QPushButton("Update soon")

        for b in (self.btn_squat, self.btn_plank, self.btn_pushup, self.btn_soon):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.btn_squat.clicked.connect(self._go_squat)
        self.btn_plank.setEnabled(False)
        self.btn_pushup.setEnabled(False)
        self.btn_soon.setEnabled(False)

        for btn in (self.btn_squat, self.btn_plank, self.btn_pushup, self.btn_soon):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFixedHeight(60)

        top = QHBoxLayout()
        top.addWidget(self.btn_squat, 1)
        top.addWidget(self.btn_plank, 1)
        top.addWidget(self.btn_pushup, 1)
        top.addWidget(self.btn_soon, 1)

        center = QLabel("운동을 선택하세요")
        center.setAlignment(Qt.AlignCenter)

        root = QVBoxLayout(self)
        root.addLayout(top)   
        root.addWidget(center, 1)  

    def _go_squat(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate("squat")
