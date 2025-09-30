# app/views/enroll_page.py
import cv2
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from core.page_base import PageBase

class EnrollPage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("EnrollPage")

        self.title = QLabel("회원가입 - 얼굴 등록")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")

        self.input_name = QLineEdit(self)
        self.input_name.setPlaceholderText("이름을 입력하세요")
        self.btn_start = QPushButton("등록 시작")
        self.btn_cancel = QPushButton("취소")

        top = QHBoxLayout()
        top.addWidget(self.input_name)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_cancel)

        self.video = QLabel("camera"); self.video.setMinimumSize(480, 360)
        self.info  = QLabel("-"); self.info.setAlignment(Qt.AlignCenter)

        grp = QGroupBox("카메라")
        gv = QVBoxLayout(); gv.addWidget(self.video); gv.addWidget(self.info)
        grp.setLayout(gv)

        root = QVBoxLayout(self)
        root.addWidget(self.title)
        root.addLayout(top)
        root.addWidget(grp)

        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)
        self.collecting = False
        self.collected = []
        self.target_n = 20

        self.btn_start.clicked.connect(self._start_collect)
        self.btn_cancel.clicked.connect(self._cancel)

    # 얼굴 등록은 front 카메라만 사용
    def on_enter(self, ctx):
        self.ctx = ctx
        self.ctx.cam.set_process("front", None)  # 원본 프레임 사용
        self.ctx.cam.start("front")
        self.timer.start(30)

    def on_leave(self, ctx):
        self.timer.stop()
        self.ctx.cam.stop("front")
        self.collecting = False
        self.collected.clear()

    def _start_collect(self):
        name = self.input_name.text().strip()
        if not name:
            QMessageBox.warning(self, "입력 오류", "이름을 입력해주세요.")
            return
        self.collecting = True
        self.collected.clear()
        self.info.setText("정면을 바라보고 자연스럽게 움직여 주세요")

    def _cancel(self):
        self.collecting = False
        self.collected.clear()
        self._goto("start")

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

        # 미리보기
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(self.video.width(), self.video.height(), Qt.KeepAspectRatio))

        if not self.collecting:
            return

        # 얼굴 임베딩 추출
        emb = self.ctx.face.detect_and_embed(frame)
        if emb is not None:
            self.collected.append(emb)

        self.info.setText(f"수집: {len(self.collected)} / {self.target_n}")

        if len(self.collected) >= self.target_n:
            # 저장
            name = self.input_name.text().strip()
            try:
                self.ctx.face.add_user_samples(name, self.collected)
                QMessageBox.information(self, "완료", f"{name} 등록이 완료되었습니다.")
                self.collecting = False
                self.collected.clear()
                self._goto("start")  # 혹은 login/select로 이동
            except Exception as e:
                QMessageBox.critical(self, "저장 실패", str(e))
                self.collecting = False
                self.collected.clear()
