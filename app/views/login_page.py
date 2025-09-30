# views/login_page.py
import cv2
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from core.page_base import PageBase

class LoginPage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("LoginPage")

        self.title = QLabel("로그인")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(800, 450)
        self.video.setStyleSheet("background: transparent; margin:0; padding:0;")
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        top = QWidget(self)
        top_v = QVBoxLayout(top)
        top_v.setContentsMargins(0, 0, 0, 0)
        top_v.addWidget(self.video, 1)

        self.info  = QLabel("카메라를 정면으로 응시해 주세요")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setStyleSheet("font-size: 30px; color: #ddd;")

        self.btn_try   = QPushButton("인식 시작")
        self.btn_enroll= QPushButton("회원가입")
        self.btn_back  = QPushButton("처음으로")

        # 버튼 사이 간격과 정렬
        btns = QHBoxLayout()
        btns.setSpacing(12)
        btns.setContentsMargins(0, 0, 0, 0)
        btns.addStretch(1)
        btns.addWidget(self.btn_try)
        btns.addWidget(self.btn_enroll)
        btns.addWidget(self.btn_back)
        btns.addStretch(1)

        bottom = QWidget(self)
        bottom_v = QVBoxLayout(bottom)
        bottom_v.setContentsMargins(16, 8, 16, 16)  # 하단 여백 약간
        bottom_v.setSpacing(10)
        bottom_v.addWidget(self.info)
        bottom_v.addLayout(btns)

        # ---- 루트 레이아웃: 상단 2, 하단 1 (즉, 2/3 : 1/3)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self.title)
        root.addWidget(top, 2)      # stretch 2
        root.addWidget(bottom, 1)   # stretch 1

        # ---- 시그널/타이머
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

        # 라벨 크기에 맞춰 비율 유지 확대
        target_w = max(1, self.video.width())
        target_h = max(1, self.video.height())
        self.video.setPixmap(
            QPixmap.fromImage(qimg).scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _recognize(self):
        frame = self.ctx.cam.frame("front")
        if frame is None:
            QMessageBox.warning(self, "오류", "카메라 프레임이 없습니다.")
            return

        emb = self.ctx.face.detect_and_embed(frame)
        if emb is None:
            self.info.setText("얼굴을 찾지 못했습니다. 다시 시도해 주세요.")
            return

        name, score = self.ctx.face.match(emb, threshold=0.50)
        if name:
            with self.ctx.SessionLocal() as s:
                from db.models import User
                user = s.query(User).filter_by(name=name).one_or_none()
                if user:
                    self.ctx.set_current_user(user.id, user.name)

            QMessageBox.information(self, "환영합니다", f"{name} 님, 인식되었습니다. (sim={score:.3f})")
            self._goto("select")
        else:
            self.info.setText(f"등록되지 않은 얼굴입니다. (최대 유사도 {score:.3f})")
