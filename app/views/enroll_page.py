# views/enroll_page.py
import os
import cv2

from core.page_base import PageBase
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QPainter, QColor, QImage, QGuiApplication
from PySide6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QStackedLayout, QSizePolicy, QFrame, QGraphicsDropShadowEffect,
    QProgressBar, QMessageBox
)

def show_keyboard():
    QGuiApplication.inputMethod().show()

def hide_keyboard():
    QGuiApplication.inputMethod().hide()

class ImageBadge(QLabel):
    def __init__(
        self,
        image_path: str,
        bg: str = "#ffffff",
        diameter: int = 96,
        inner_padding: int = 2,
        parent=None,
    ):
        super().__init__(parent)
        self.bg = bg
        self.diameter = max(1, diameter)
        self.inner_padding = max(0, inner_padding)
        self.image = QPixmap(image_path) if os.path.exists(image_path) else None

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(self.diameter, self.diameter)

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        return self.size()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        r = self.rect()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.bg))
        p.drawEllipse(r.adjusted(0, 0, -1, -1))

        if self.image and not self.image.isNull():
            icon_max = self.diameter - (self.inner_padding * 2)
            icon = self.image.scaled(icon_max, icon_max, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (r.width() - icon.width()) // 2
            y = (r.height() - icon.height()) // 2
            p.drawPixmap(x, y, icon)

        p.end()


class GlassCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GlassCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            #GlassCard {
                border-radius: 20px;
                background: rgba(0,0,0,0.45);
                border: 1px solid rgba(255,255,255,0.20);
            }
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 170))
        self.setGraphicsEffect(shadow)

        self._inner = QWidget(self)
        self._inner.setObjectName("CardInner")
        self._inner.setAttribute(Qt.WA_TranslucentBackground)
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(28, 28, 28, 28)
        self._inner_layout.setSpacing(16)

    def inner_layout(self) -> QVBoxLayout:
        return self._inner_layout

    def addWidget(self, w: QWidget):
        self._inner_layout.addWidget(w)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._inner.setGeometry(self.rect())

class TouchLineEdit(QLineEdit):
    def focusInEvent(self, e):
        super().focusInEvent(e)
        show_keyboard()

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        hide_keyboard()

def make_step_indicator(step1_active: bool, step2_active: bool) -> QWidget:
    wrap = QWidget()
    h = QHBoxLayout(wrap)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(10)

    def pill(text: str, active: bool) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(32)
        lbl.setMinimumWidth(120)
        lbl.setStyleSheet(
            f"""
            QLabel {{
                padding: 6px 16px;
                border-radius: 16px;
                font-size: 14px;
                font-weight: 600;
                border: 1px solid {"rgba(255,255,255,0.15)" if active else "rgba(255,255,255,0.25)"};
                background: {"qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2daaff, stop:1 #157bff)" if active else "rgba(255,255,255,0.12)"};
                color: {"white" if active else "rgba(255,255,255,0.75)"};
            }}
            """
        )
        return lbl

    h.addWidget(pill("①  정보 입력", step1_active))
    h.addWidget(pill("②  얼굴 인식", step2_active))
    h.addStretch(1)
    return wrap

class BaseCardView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.card = GlassCard(self)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        wrap = QWidget(self)
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addStretch(1)
        wl.addWidget(self.card, 0, Qt.AlignHCenter)
        wl.addStretch(1)
        outer.addWidget(wrap)

    def _sync_card_size(self):
        w = max(400, int(self.width() * 0.90))
        h = max(300, int(self.height() * 0.90))
        self.card.setFixedSize(w, h)

    def showEvent(self, _):
        self._sync_card_size()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_card_size()

class UserInfoView(BaseCardView):
    next_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        cl = self.card.inner_layout()
        cl.setSpacing(12)

        badge = ImageBadge("app/assets/icon_register.png", bg="#FFFFFF", diameter=96, inner_padding=2, parent=self)
        title = QLabel("회원가입")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:34px; font-weight:800; color:white;")

        top = QWidget(self.card)
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        top_l.setSpacing(10)
        top_l.addWidget(badge, 0, Qt.AlignHCenter)
        top_l.addWidget(title, 0, Qt.AlignHCenter)

        stepper = make_step_indicator(True, False)

        subtitle = QLabel("회원님의 성함을 입력해주세요")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color:rgba(255,255,255,0.85); font-size:14px;")

        self.name_edit = TouchLineEdit()
        self.name_edit.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.name_edit.setPlaceholderText("이름을 입력하세요")
        self.name_edit.setFixedHeight(44)
        self.name_edit.setMaxLength(32)
        self.name_edit.setAlignment(Qt.AlignCenter)
        self.name_edit.setFixedWidth(420)
        self.name_edit.setStyleSheet("""
            QLineEdit {
                font-size:16px; border-radius:12px; padding:8px 14px;
                color:white; background:rgba(255,255,255,0.10);
                border:1px solid rgba(255,255,255,0.28);
            }
            QLineEdit:focus {
                border-color:rgba(255,255,255,0.55);
                background:rgba(255,255,255,0.16);
            }
        """)

        self.next_btn = QPushButton("다음 단계")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.setFixedSize(420, 44)
        self.next_btn.setStyleSheet("""
            QPushButton {
                font-weight:800; border-radius:12px; border:1px solid rgba(255,255,255,0.15);
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2daaff, stop:1 #157bff);
                color:white;
            }
            QPushButton:hover {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #35b3ff, stop:1 #1b86ff);
            }
            QPushButton:pressed {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #209cf2, stop:1 #146fe0);
            }
        """)
        self.next_btn.clicked.connect(self._emit_next)

        cl.addWidget(top)
        cl.addWidget(stepper)
        cl.addSpacing(8)
        cl.addWidget(subtitle)
        cl.addSpacing(6)
        cl.addWidget(self.name_edit, 0, Qt.AlignHCenter)
        cl.addSpacing(12)
        cl.addWidget(self.next_btn, 0, Qt.AlignHCenter)
        cl.addStretch(1)

    def _emit_next(self):
        hide_keyboard()
        self.next_requested.emit(self.name_edit.text().strip())

class FaceScanView(BaseCardView):
    finished = Signal()

    def __init__(self, parent=None, target_n: int = 20):
        super().__init__(parent)
        self.target_n = target_n
        self.collecting = False
        self.collected = []
        self.name = ""
        self.ctx = None

        cl = self.card.inner_layout()
        cl.setSpacing(12)

        badge = ImageBadge("app/assets/icon_register.png", bg="#2b98ff", diameter=72, inner_padding=2, parent=self)
        title = QLabel("얼굴 인식")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:28px; font-weight:800; color:white;")

        top = QWidget(self.card)
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)
        tl.addWidget(badge, 0, Qt.AlignHCenter)
        tl.addWidget(title, 0, Qt.AlignHCenter)

        stepper = make_step_indicator(False, True)

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setMinimumSize(960, 540)
        self.video.setStyleSheet("background:rgba(255,255,255,0.08); border-radius:12px;")

        self.info = QLabel("정면을 바라보고 자연스럽게 움직여 주세요")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setStyleSheet("font-size:16px; color:rgba(255,255,255,0.90);")

        self.bar = QProgressBar()
        self.bar.setRange(0, self.target_n)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(22)
        self.bar.setStyleSheet("""
            QProgressBar {
                border:1px solid rgba(255,255,255,0.35); border-radius:11px;
                background:rgba(255,255,255,0.08); padding:2px;
            }
            QProgressBar::chunk {
                border-radius:11px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ADD8E6, stop:1 #87CEFA);
            }
        """)

        self.start_btn = QPushButton("얼굴 등록 시작")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setFixedSize(220, 44)
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-weight:800; border-radius:12px; border:1px solid rgba(255,255,255,0.15);
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2daaff, stop:1 #157bff);
                color:white;
            }
            QPushButton:disabled { opacity:.7; }
        """)
        self.start_btn.clicked.connect(self._start_collection)

        cl.addWidget(top)
        cl.addWidget(stepper)
        cl.addSpacing(8)
        cl.addWidget(self.video, 1)
        cl.addSpacing(6)
        cl.addWidget(self.info)
        cl.addWidget(self.bar, 0, Qt.AlignHCenter)
        cl.addSpacing(8)
        cl.addWidget(self.start_btn, 0, Qt.AlignHCenter)
        cl.addStretch(0)

        self._last_qimg = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def begin(self, ctx, name: str):
        self.ctx = ctx
        self.name = name.strip()
        self.collected.clear()
        self.bar.setMaximum(self.target_n)
        self.bar.setValue(0)
        self.info.setText("카메라 앞에 얼굴을 위치시켜 주세요.")
        self.start_btn.setText("얼굴 등록 시작")
        self.start_btn.setEnabled(True)

        if self.ctx:
            try:
                self.ctx.cam.start()
            except Exception as e:
                print("[EnrollPage] Camera start failed:", e)
        self.collecting = False
        self._timer.start(30)

    def _start_collection(self):
        if not self.ctx:
            return
        hide_keyboard()
        self.collecting = True
        self.collected.clear()
        self.bar.setValue(0)
        self.info.setText("정면을 바라보고 자연스럽게 움직여 주세요")
        self.start_btn.setText("수집 중...")
        self.start_btn.setEnabled(False)

    def end(self):
        hide_keyboard()
        self.collecting = False
        self._timer.stop()
        if self.ctx:
            try:
                self.ctx.cam.stop()  
            except Exception:
                pass
        self.start_btn.setText("얼굴 등록 시작")
        self.start_btn.setEnabled(True)

    def _render_frame(self, qimg: QImage):
        self._last_qimg = qimg
        tw, th = max(1, self.video.width()), max(1, self.video.height())
        p = QPixmap.fromImage(qimg).scaled(
            tw, th, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
        )
        if p.width() > tw or p.height() > th:
            x = max(0, (p.width() - tw) // 2)
            y = max(0, (p.height() - th) // 2)
            p = p.copy(x, y, tw, th)
        self.video.setPixmap(p)

    def _tick(self):
        if not self.ctx:
            return
        frame = self.ctx.cam.frame()  
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._render_frame(qimg)

        if not self.collecting:
            return

        emb = self.ctx.face.detect_and_embed(frame)
        if emb is not None:
            self.collected.append(emb)
            self.bar.setValue(len(self.collected))

        if len(self.collected) >= self.target_n:
            try:
                self.ctx.face.add_user_samples(self.name, self.collected)
                self.end()
                QMessageBox.information(self, "완료", f"{self.name} 등록이 완료되었습니다.")
            except Exception as e:
                self.end()
                QMessageBox.critical(self, "저장 실패", str(e))
            finally:
                self.finished.emit()

class EnrollPage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("EnrollPage")
        self.ctx = None
        self._current_name = ""

        self._bg_label = QLabel(self)
        self._bg_label.setScaledContents(True)
        self._bg_overlay = QLabel(self)
        self._bg_overlay.setStyleSheet("background:rgba(0,0,0,0.55);")
        self._bg_label.lower()
        self._bg_overlay.raise_()

        bg_path = os.path.join(os.getcwd(), "app/assets", "bg_gym.jpg")
        if os.path.exists(bg_path):
            pix = QPixmap(bg_path)
            if not pix.isNull():
                self._bg_pix = pix
                self._bg_label.setPixmap(self._bg_pix)
            else:
                self._bg_label.setStyleSheet("background:#0a0f19;")
        else:
            self._bg_label.setStyleSheet(
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0e1a2b, stop:1 #0a0f19);"
            )

        self._stack = QStackedLayout()
        self.user_info_view = UserInfoView()
        self.face_scan_view = FaceScanView()

        self.user_info_view.next_requested.connect(self._go_face_scan)
        self.face_scan_view.finished.connect(self._go_start)

        self._stack.addWidget(self.user_info_view)
        self._stack.addWidget(self.face_scan_view)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addLayout(self._stack)

        self.setStyleSheet("* { color:white; }")
        self._stack.setCurrentWidget(self.user_info_view)

    def on_enter(self, ctx):
        self.ctx = ctx
        self._reset_ui()

    def on_leave(self, _):
        hide_keyboard()
        try:
            self.face_scan_view.end()
        except Exception:
            pass

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_background()

    def _sync_background(self):
        self._bg_label.setGeometry(self.rect())
        self._bg_overlay.setGeometry(self.rect())
        if getattr(self, "_bg_pix", None):
            self._bg_label.setPixmap(
                self._bg_pix.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )

    def _go_face_scan(self, name: str):
        name = (name or "").strip()
        if not name:
            self.user_info_view.name_edit.setPlaceholderText("이름을 입력해주세요")
            return
        self._current_name = name
        hide_keyboard()
        self._stack.setCurrentWidget(self.face_scan_view)
        self.face_scan_view.begin(self.ctx, name)

    def _go_start(self):
        hide_keyboard()
        self.ctx.router.navigate("start")

    def _reset_ui(self):
        hide_keyboard()
        self._current_name = ""
        self.user_info_view.name_edit.clear()
        self.user_info_view.next_btn.setEnabled(True)
        try:
            self.face_scan_view.end()
        except Exception:
            pass
        self.face_scan_view.collected.clear()
        self.face_scan_view.bar.setValue(0)
        self.face_scan_view.info.setText("카메라 앞에 얼굴을 위치시켜 주세요.")
        self.face_scan_view.start_btn.setText("얼굴 등록 시작")
        self.face_scan_view.start_btn.setEnabled(True)
        self.face_scan_view.video.clear()
        self.face_scan_view._last_qimg = None
        self._stack.setCurrentWidget(self.user_info_view)
