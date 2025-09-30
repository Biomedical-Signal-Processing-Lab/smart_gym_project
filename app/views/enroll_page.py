# views/enroll_page.py
import cv2
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout,
    QMessageBox, QSizePolicy, QProgressBar
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from core.page_base import PageBase

class EnrollPage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("EnrollPage")

        self.collecting = False
        self.collected = []
        self.target_n = 20

        # ---- 타이틀
        self.title = QLabel("회원가입")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 22px; font-weight: 700;")

        # ---- 카메라
        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(800, 450)
        self.video.setStyleSheet("background: transparent; margin:0; padding:0;")
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored) 
        self._last_qimg = None  

        top = QWidget(self)
        top_v = QVBoxLayout(top)
        top_v.setContentsMargins(0, 0, 0, 0)
        top_v.addWidget(self.video, 1)

        # 안내 + 이름입력 + 버튼
        self.info = QLabel("이름을 입력하고 등록시작 버튼을 눌러주세요.")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setStyleSheet("font-size: 20px; color: #ddd;")

        self.bar = QProgressBar()
        self.bar.setRange(0, self.target_n)      
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(20)
        self.bar.setFixedWidth(400)   
        self.bar.setAlignment(Qt.AlignCenter)
        self.bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid rgba(255,255,255,0.35);
                border-radius: 10px;
                background: rgba(255,255,255,0.08);
                text-align: center;
                color: white;
                padding: 2px;
            }
            QProgressBar::chunk {
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #ADD8E6,   
                                            stop:1 #87CEFA);
            }
        """)

        self.input_name = QLineEdit(self)
        self.input_name.setPlaceholderText("이름을 입력하세요")
        self.input_name.setAlignment(Qt.AlignCenter)
        self.input_name.setFixedHeight(40)
        self.input_name.setFixedWidth(400)   
        self.input_name.setAlignment(Qt.AlignCenter)
        self.input_name.setStyleSheet("""
            QLineEdit {
                font-size: 16px; padding: 6px 12px;
                border-radius: 10px; border: 1px solid rgba(255,255,255,0.35);
                background: rgba(255,255,255,0.10); color: white;
            }
            QLineEdit:focus { border-color: rgba(255,255,255,0.65); }
        """)

        self.btn_start = QPushButton("등록 시작")
        self.btn_cancel = QPushButton("취소")

        btn_qss = """
            QPushButton {
                padding: 10px 24px; font-size: 16px; font-weight: 700;
                border-radius: 12px; border: 1px solid rgba(255,255,255,0.35);
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 rgba(255,255,255,0.22),
                                            stop:1 rgba(255,255,255,0.08));
                color: white;
            }
            QPushButton:hover {
                background: rgba(255,255,255,0.25);
                border-color: rgba(255,255,255,0.55);
            }
            QPushButton:pressed {
                background: rgba(255,255,255,0.18);
                border-color: rgba(255,255,255,0.35);
            }
        """
        self.btn_start.setStyleSheet(btn_qss)
        self.btn_cancel.setStyleSheet(btn_qss)

        btns = QHBoxLayout()
        btns.setSpacing(12)
        btns.setContentsMargins(0, 0, 0, 0)
        btns.addStretch(1)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_cancel)
        btns.addStretch(1)

        bottom = QWidget(self)
        bottom_v = QVBoxLayout(bottom)
        bottom_v.setContentsMargins(16, 8, 16, 16)
        bottom_v.setSpacing(10)
        bottom_v.addWidget(self.info,       0, Qt.AlignHCenter)
        bottom_v.addWidget(self.bar,        0, Qt.AlignHCenter)
        bottom_v.addWidget(self.input_name, 0, Qt.AlignHCenter)
        bottom_v.addLayout(btns)

        # ---- 루트 레이아웃
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self.title)
        root.addWidget(top, 2)       
        root.addWidget(bottom, 1)    

        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)

        self.btn_start.clicked.connect(self._start_collect)
        self.btn_cancel.clicked.connect(self._cancel)

    # 얼굴 등록은 front 카메라만 사용
    def on_enter(self, ctx):
        self.ctx = ctx
        self.ctx.cam.set_process("front", None)  
        self.ctx.cam.start("front")
        self.timer.start(30)

    def on_leave(self, ctx):
        self.timer.stop()
        self.ctx.cam.stop("front")
        self.collecting = False
        self.collected.clear()

    # 공통 렌더러: 여백 없이 꽉 채우는 센터-크롭
    def _render_frame(self, qimg: QImage):
        self._last_qimg = qimg
        tw, th = max(1, self.video.width()), max(1, self.video.height())

        p = QPixmap.fromImage(qimg).scaled(
            tw, th,
            Qt.KeepAspectRatioByExpanding,   
            Qt.SmoothTransformation
        )
        if p.width() > tw or p.height() > th:
            x = max(0, (p.width() - tw) // 2)
            y = max(0, (p.height() - th) // 2)
            p = p.copy(x, y, tw, th)

        self.video.setPixmap(p)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._last_qimg is not None:
            self._render_frame(self._last_qimg)

    def _start_collect(self):
        name = self.input_name.text().strip()
        if not name:
            QMessageBox.warning(self, "입력 오류", "이름을 입력해주세요.")
            return
        self.collecting = True
        self.collected.clear()
        self.bar.setRange(0, self.target_n)  
        self.bar.setValue(0)
        self.info.setText("정면을 바라보고 자연스럽게 움직여 주세요")
        self.input_name.setEnabled(False)
        self.btn_start.setEnabled(False)

    def _cancel(self):
        self.collecting = False
        self.collected.clear()
        self.bar.setValue(0)
        self.input_name.setEnabled(True)
        self.btn_start.setEnabled(True)
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

        # 미리보기 (센터 크롭)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self._render_frame(qimg)

        if not self.collecting:
            return

        # 얼굴 임베딩 추출 & 수집
        emb = self.ctx.face.detect_and_embed(frame)
        if emb is not None:
            self.collected.append(emb)

        cur = len(self.collected)
        self.bar.setValue(cur)

        if len(self.collected) >= self.target_n:
            name = self.input_name.text().strip()
            try:
                self.ctx.face.add_user_samples(name, self.collected)
                QMessageBox.information(self, "완료", f"{name} 등록이 완료되었습니다.")
                self.collecting = False
                self.collected.clear()
                self.bar.setValue(0)
                self.input_name.setEnabled(True)
                self.btn_start.setEnabled(True)
                self._goto("start")
            except Exception as e:
                QMessageBox.critical(self, "저장 실패", str(e))
                self.collecting = False
                self.collected.clear()
                self.bar.setValue(0)
                self.input_name.setEnabled(True)
                self.btn_start.setEnabled(True)

