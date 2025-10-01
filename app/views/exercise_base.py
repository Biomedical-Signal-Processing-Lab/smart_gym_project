# views/exercise_base.py
import time, cv2
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QPushButton
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage
from core.page_base import PageBase
from core.mp_manager import PoseProcessor
from ui.overlay_painter import VideoCanvas

class ExerciseBasePage(PageBase):
    PAGE_FPS_MS = 33  

    def __init__(self, title: str = "exercise"):
        super().__init__()
        self.proc = PoseProcessor(model_complexity=1, name=f"pose_{title}")

        self.canvas = VideoCanvas()
        self.canvas._video.setMinimumSize(400, 400)

        self.chk_skel = QCheckBox("Skeleton")
        self.chk_skel.setChecked(True)
        self.chk_skel.stateChanged.connect(lambda _: self.proc.set_draw_landmarks(self.chk_skel.isChecked()))

        self.lbl_bottom = QLabel("-")
        self.lbl_bottom.setAlignment(Qt.AlignCenter)

        self.btn_end = QPushButton("운동 종료")
        self.btn_end.clicked.connect(self._end_workout)

        ctl = QHBoxLayout()
        ctl.addStretch(1); ctl.addWidget(self.chk_skel); ctl.addStretch(1)

        bottom = QVBoxLayout()
        bottom.addLayout(ctl)
        bottom.addWidget(self.lbl_bottom)

        end_row = QHBoxLayout()
        end_row.addStretch(1); end_row.addWidget(self.btn_end); end_row.addStretch(1)
        bottom.addLayout(end_row)

        root = QVBoxLayout(self)
        root.addWidget(self.canvas, stretch=2)
        wrap = QWidget(); wrap.setLayout(bottom)
        root.addWidget(wrap, stretch=1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self._session_started_ts = None
        self._score_sum = 0.0
        self._score_n   = 0

    def on_enter(self, ctx):
        self.ctx = ctx
        self._session_started_ts = time.time()
        self._score_sum = 0.0; self._score_n = 0
        self._hook_reset()

        self.ctx.cam.set_process(self.proc)
        self.proc.set_draw_landmarks(self.chk_skel.isChecked())

        self.canvas.clear_overlays()
        self.build_overlays(self.canvas)

        self.ctx.cam.start()
        if self.timer.isActive(): self.timer.stop()
        self.timer.start(self.PAGE_FPS_MS)

    def on_leave(self, ctx):
        if self.timer.isActive(): self.timer.stop()
        ctx.cam.stop()
        self.canvas.clear_overlays()

    def _tick(self):
        frame = self.ctx.cam.frame()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.canvas.set_frame(qimg)

        meta = self.ctx.cam.meta() or {}
        self.on_meta(meta)

        self.lbl_bottom.setText(self.format_bottom_text(meta))

    def build_overlays(self, canvas: VideoCanvas):
        """영상 위 UI를 배치(버튼/게이지/라벨 등). anchor 사용 가능."""
        pass

    def on_meta(self, meta: dict):
        """운동별 상태머신/스코어링 업데이트."""
        pass

    def format_bottom_text(self, meta: dict) -> str:
        """하단 표시 텍스트 반환."""
        return "-"

    def _hook_reset(self):
        """운동별 상태 초기화(필요 시 오버라이드)."""
        pass

    def _end_workout(self):
        ended_at = time.time()
        dur = 0.0 if self._session_started_ts is None else (ended_at - self._session_started_ts)
        avg = (self._score_sum / self._score_n) if self._score_n > 0 else 0.0

        summary = {
            "exercise": self.__class__.__name__.replace("Page","").lower(),
            "reps": getattr(self, "reps", 0),
            "avg_score": round(avg, 1),
            "duration_sec": int(dur),
            "started_at": self._session_started_ts,
            "ended_at": ended_at,
        }
        try:
            if hasattr(self.ctx, "save_workout_session"):
                self.ctx.save_workout_session(summary)
        except Exception as e:
            print(f"[WARN] workout save failed: {e}")

        self.timer.stop()
        try: self.ctx.cam.stop()
        except Exception: pass

        if hasattr(self.ctx, "goto_summary"):
            self.ctx.goto_summary(summary)
