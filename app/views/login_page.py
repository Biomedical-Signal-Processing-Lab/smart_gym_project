# app/views/login_page.py
import cv2
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from core.page_base import PageBase

class LoginPage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("LoginPage")

        self.title = QLabel("로그인 - 얼굴 인식")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")

        self.video = QLabel("camera"); self.video.setMinimumSize(480, 360)
        self.info  = QLabel("카메라를 응시해 주세요"); self.info.setAlignment(Qt.AlignCenter)

        self.btn_try = QPushButton("인식 시도")
        self.btn_back = QPushButton("처음으로")
        self.btn_enroll = QPushButton("회원가입")

        b = QHBoxLayout(); b.addStretch(1); b.addWidget(self.btn_try); b.addWidget(self.btn_enroll); b.addWidget(self.btn_back); b.addStretch(1)

        grp = QGroupBox("카메라")
        gv = QVBoxLayout(); gv.addWidget(self.video); gv.addWidget(self.info); grp.setLayout(gv)

        root = QVBoxLayout(self)
        root.addWidget(self.title)
        root.addWidget(grp)
        root.addLayout(b)

        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)
        self.btn_try.clicked.connect(self._recognize)
        self.btn_back.clicked.connect(lambda: self._goto("start"))
        self.btn_enroll.clicked.connect(lambda: self._goto("enroll"))

    def on_enter(self, ctx):
        self.ctx = ctx
        self.ctx.cam.set_process("front", None)
        self.ctx.cam.start("front")
        self.timer.start(30)

    def on_leave(self, ctx):
        self.timer.stop()
        self.ctx.cam.stop("front")

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    def _tick(self):
        frame = self.ctx.cam.frame("front")
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(self.video.width(), self.video.height(), Qt.KeepAspectRatio))

    def _recognize(self):
        frame = self.ctx.cam.frame("front")
        if frame is None:
            QMessageBox.warning(self, "오류", "카메라 프레임이 없습니다.")
            return

        emb = self.ctx.face.detect_and_embed(frame)
        if emb is None:
            self.info.setText("얼굴을 찾지 못했습니다. 다시 시도해 주세요.")
            return

        name, score = self.ctx.face.match(emb, threshold=0.50)  # 초기엔 0.40 정도로
        if name:
            QMessageBox.information(self, "환영합니다", f"{name} 님, 인식되었습니다. (sim={score:.3f})")
            self._goto("select")
        else:
            self.info.setText(f"등록되지 않은 얼굴입니다. (최대 유사도 {score:.3f})")
