from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSizePolicy, QStackedLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from core.page_base import PageBase

class StartPage(PageBase):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("background: black;")
        self.setMinimumSize(1, 1)
        self._last_image = None

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(1000)
        self._auto_timer.timeout.connect(self._auto_login_tick)
        self._hit_consecutive = 0
        self._need_hits = 2
        self._th_sim = 0.50

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self); self.audio.setVolume(0.0)
        self.player.setAudioOutput(self.audio)

        self.sink = QVideoSink(self)
        self.player.setVideoOutput(self.sink)
        self.sink.videoFrameChanged.connect(self._on_frame)

        title = QLabel("자세어때"); title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: black;
            font-size: 120px;
            font-weight: 1000;
        """)

        subtitle = QLabel("")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            color: white;
            font-size: 25px;
            font-weight: 400;
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

        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: white; font-size: 18px;")
        ov.addWidget(self.lbl_status)

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

    def _go_login(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"): router = router.parent()
        if router: router.navigate("login")

    def _go_enroll(self):
        router = self.parent()
        while router and not hasattr(router, "navigate"): router = router.parent()
        if router: router.navigate("enroll")

    def on_enter(self, ctx):
        try:
            self.player.play()
        except Exception:
            pass

        self.ctx = ctx

        try:
            self.ctx.cam.start()   
        except Exception as e:
            print("[StartPage] Camera start failed:", e)

        self._hit_consecutive = 0
        self.lbl_status.setText("카메라 준비 중… 정면을 바라봐 주세요.")
        self._auto_timer.start()

    def on_leave(self, ctx):
        try: self.player.pause()
        except Exception: pass
        self._auto_timer.stop()

    def _auto_login_tick(self):
        frame = None
        try:
            frame = self.ctx.cam.frame()  
            if frame is None:
                self.lbl_status.setText("카메라 준비 중…")
                return
        except Exception:
            self.lbl_status.setText("카메라 오류. 장치를 확인해 주세요.")
            return

        emb = self.ctx.face.detect_and_embed(frame)
        if emb is None:
            self._hit_consecutive = 0
            self.lbl_status.setText("얼굴을 화면 중앙에 맞추고 정면을 바라봐 주세요.")
            return

        name, sim = self.ctx.face.match(emb, threshold=self._th_sim)
        if name:
            self._hit_consecutive += 1
            self.lbl_status.setText(f"{name} 님으로 인식 중… ({self._hit_consecutive}/{self._need_hits})")
            if self._hit_consecutive >= self._need_hits:
                try:
                    from db.models import User
                    with self.ctx.SessionLocal() as s:
                        user = s.query(User).filter_by(name=name).one_or_none()
                        if user:
                            self.ctx.set_current_user(user.id, user.name)
                except Exception:
                    pass
                self._auto_timer.stop()
                self._goto("guide")
        else:
            self._hit_consecutive = 0
            self.lbl_status.setText("등록되지 않은 얼굴이에요. ‘회원가입’을 눌러 등록해 주세요.")

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"): router = router.parent()
        if router: router.navigate(page)
