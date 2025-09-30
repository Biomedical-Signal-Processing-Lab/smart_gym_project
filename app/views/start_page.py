# views/start_page.py 
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSizePolicy, QStackedLayout
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from core.page_base import PageBase
import os

class StartPage(PageBase):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("background: black;")
        self.setMinimumSize(1, 1)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self); self.audio.setVolume(0.0)
        self.player.setAudioOutput(self.audio)

        self.sink = QVideoSink(self)
        self.player.setVideoOutput(self.sink)
        self.sink.videoFrameChanged.connect(self._on_frame)

        video_path = os.path.join(os.path.dirname(__file__), "..", "assets", "start.mov")
        self.player.setSource(QUrl.fromLocalFile(os.path.abspath(video_path)))
        try:
            self.player.setLoops(QMediaPlayer.Infinite)
        except Exception:
            self.player.mediaStatusChanged.connect(self._loop_if_needed)

        title = QLabel("헬스왕"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: black;
            font-size: 120px;
            font-weight: 1000;   
            text-shadow: 0 2px 6px rgba(0,0,0,0.6);
        """)

        subtitle = QLabel("")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            color: white;
            font-size: 25px;
            font-weight: 400;
            text-shadow: 0 1px 4px rgba(0,0,0,0.5);
        """)

        self.btn_login  = QPushButton("로그인")
        self.btn_signup = QPushButton("회원가입")

        for b in (self.btn_login, self.btn_signup):
            b.setFixedHeight(45)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            b.setStyleSheet("""
                QPushButton {
                    padding: 8px 24px; font-size: 25px; background: rgba(255,255,255,0.9);
                    color: black; border: none; border-radius: 8px;
                }
                QPushButton:hover { background: rgba(255,255,255,1.0); }
            """)

        self.btn_login.clicked.connect(self._go_login)
        self.btn_signup.clicked.connect(self._go_enroll)

        overlay = QWidget(self)
        ov = QVBoxLayout(overlay)
        ov.setContentsMargins(24, 24, 24, 24)
        ov.setSpacing(12)

        ov.addSpacing(120)     
        ov.addWidget(title)
        ov.addWidget(subtitle)
        ov.addStretch(20)

        h = QHBoxLayout()
        h.addStretch(1); h.addWidget(self.btn_login); h.addSpacing(16)
        h.addWidget(self.btn_signup); h.addStretch(1); ov.addLayout(h); ov.addStretch(3)
        overlay.setStyleSheet("background: transparent;")

        stack = QStackedLayout(self)
        stack.setStackingMode(QStackedLayout.StackAll)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        stack.addWidget(self.video_label)
        stack.addWidget(overlay)
        overlay.raise_()

        self._last_image = None

    def _on_frame(self, frame):
        img = frame.toImage()
        if img.isNull():
            return
        self._last_image = img
        scaled = QPixmap.fromImage(img).scaled(
            self.video_label.size(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._last_image and not self._last_image.isNull():
            scaled = QPixmap.fromImage(self._last_image).scaled(
                self.video_label.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled)

    def _loop_if_needed(self, status):
        from PySide6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.EndOfMedia:
            self.player.setPosition(0); self.player.play()

    def _go_login(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"): router = router.parent()
        if router: router.navigate("login")

    def _go_enroll(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"): router = router.parent()
        if router: router.navigate("enroll")

    def on_enter(self, ctx):
        try: self.player.play()
        except Exception: pass

    def on_leave(self, ctx):
        try: self.player.pause()
        except Exception: pass
