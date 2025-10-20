from dataclasses import dataclass, field
from typing import Optional
from PySide6.QtCore import Qt, QSize, QRect, Signal, QObject
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPainterPath, QPen

# ──────────────────────────────────────────────
# 📸 영상 캔버스 (비디오 + 오버레이)
# ──────────────────────────────────────────────
class VideoCanvas(QWidget):
    _anchor_to_cell = {
        "top-left": (0,0), "top-center": (0,1), "top-right": (0,2),
        "center-left": (1,0), "center": (1,1), "center-right": (1,2),
        "bottom-left": (2,0), "bottom-center": (2,1), "bottom-right": (2,2),
    }

    def __init__(self, min_size: QSize | None = None, parent=None):
        super().__init__(parent)
        self._video = QLabel("camera: no frame", self)
        self._video.setAlignment(Qt.AlignCenter)
        if min_size:
            self._video.setMinimumSize(min_size)

        self._overlay_root = QWidget(self)
        self._overlay_root.setAttribute(Qt.WA_StyledBackground, False)
        self._overlay_root.setStyleSheet("background: transparent;")

        grid = QGridLayout(self._overlay_root)
        grid.setContentsMargins(12,12,12,12)
        grid.setSpacing(8)
        for i in range(3):
            grid.setColumnStretch(i, 1)
            grid.setRowStretch(i, 1)

        self._cells = [[QWidget(self._overlay_root) for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                v_align = Qt.AlignTop if r == 0 else (Qt.AlignVCenter if r == 1 else Qt.AlignBottom)
                grid.addWidget(self._cells[r][c], r, c, alignment=v_align)

        self._last_qimage: QImage | None = None
        self._img_w = None
        self._img_h = None
        self._fit_mode = "cover"

    def set_fit_mode(self, mode: str):
        self._fit_mode = "cover" if str(mode).lower() == "cover" else "contain"
        self._position_layers()

    def _compute_target_rect(self) -> QRect:
        return QRect(0, 0, self.width(), self.height())

    def _position_layers(self):
        rect = self._compute_target_rect()
        self._video.setGeometry(rect)
        self._overlay_root.setGeometry(rect)
        self._overlay_root.raise_()

        if self._last_qimage is not None and rect.width() > 0 and rect.height() > 0:
            aspect_flag = Qt.KeepAspectRatioByExpanding if self._fit_mode == "cover" else Qt.KeepAspectRatio
            pm = QPixmap.fromImage(self._last_qimage).scaled(
                rect.width(), rect.height(), aspect_flag, Qt.SmoothTransformation
            )
            self._video.setPixmap(pm)
            self._video.setAlignment(Qt.AlignCenter)

    def set_frame(self, qimage: QImage):
        if qimage is None or qimage.isNull():
            self._last_qimage = None
            self._img_w = self._img_h = None
            self._video.clear()
            return
        self._last_qimage = qimage.copy()
        self._img_w = self._last_qimage.width()
        self._img_h = self._last_qimage.height()
        self._position_layers()

    # ------ 오버레이 추가 ------
    def add_overlay(self, widget: QWidget, anchor: str = "top-right"):
        r, c = self._anchor_to_cell.get(anchor, (0, 2))
        cell = self._cells[r][c]
        lay = cell.layout()
        if lay is None:
            lay = QHBoxLayout(cell)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)

        if anchor.endswith("left"):
            h_align = Qt.AlignLeft
        elif anchor.endswith("center"):
            h_align = Qt.AlignHCenter
        else:
            h_align = Qt.AlignRight

        lay.addWidget(widget, 0, h_align)
        widget.show()

    def clear_overlays(self):
        for row in self._cells:
            for cell in row:
                lay = cell.layout()
                if lay is not None:
                    while lay.count():
                        item = lay.takeAt(0)
                        w = item.widget()
                        if w is not None:
                            w.hide()
                            w.setParent(self._overlay_root)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._position_layers()

# ──────────────────────────────────────────────
# 🧍 운동 카드 / 점수 패널 / 버튼
# ──────────────────────────────────────────────
class ExerciseCard(QWidget):
    def __init__(self, title: str = "휴식중", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background: rgba(0, 0, 0, 160);
                border-radius: 22px;
            }
            QLabel#caption {
                color: #BFC6CF; font-size: 20px; font-weight: 600; letter-spacing: 0.5px;
                background: transparent;
            }
            QLabel#titleValue {
                color: #FFFFFF; font-size: 48px; font-weight: 900;
                letter-spacing: 1px; background: transparent;
            }
            QLabel#countValue {
                color: #00E0FF; font-size: 96px; font-weight: 900;
                letter-spacing: 1px; background: transparent;
            }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(6)
        self._lbl_caption_title = QLabel("운동 종류", self)
        self._lbl_caption_title.setObjectName("caption")
        self._lbl_title_value = QLabel(title, self)
        self._lbl_title_value.setObjectName("titleValue")
        self._lbl_caption_count = QLabel("운동 횟수", self)
        self._lbl_caption_count.setObjectName("caption")
        self._lbl_count_value = QLabel("0", self)
        self._lbl_count_value.setObjectName("countValue")
        lay.addWidget(self._lbl_caption_title)
        lay.addWidget(self._lbl_title_value)
        lay.addSpacing(6)
        lay.addWidget(self._lbl_caption_count)
        lay.addWidget(self._lbl_count_value)

    def set_title(self, title: str): self._lbl_title_value.setText(title)
    def set_count(self, n: int): self._lbl_count_value.setText(str(int(n)))

class ScoreAdvicePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget { background: rgba(0, 0, 0, 160); border-radius: 22px; }
            QLabel#caption { color: #BFC6CF; font-size: 20px; font-weight: 600; }
            QLabel#score { color: #FFD166; font-size: 72px; font-weight: 900; }
            QLabel#advice { color: #FFFFFF; font-size: 22px; font-weight: 600; line-height: 130%; }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(8)
        self._lbl_caption = QLabel("평균 점수", self); self._lbl_caption.setObjectName("caption")
        self._lbl_score = QLabel("0", self); self._lbl_score.setObjectName("score")
        self._lbl_advice = QLabel("", self); self._lbl_advice.setObjectName("advice"); self._lbl_advice.setWordWrap(True)
        lay.addWidget(self._lbl_caption); lay.addWidget(self._lbl_score); lay.addSpacing(6); lay.addWidget(self._lbl_advice)
    def set_avg(self, v: float | int):
        try: v = int(round(float(v)))
        except Exception: v = 0
        self._lbl_score.setText(str(v))
    def set_advice(self, text: str): self._lbl_advice.setText(text or "")

class ActionButtons(QWidget):
    endClicked = Signal(); infoClicked = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget { background: transparent; }
            QPushButton {
                border:none; border-radius:14px; padding:14px 22px;
                font-size:20px; font-weight:600;
            }
            QPushButton#btn-info {
                background: rgba(255,255,255,0.14); color: #FFFFFF;
                border:2px solid rgba(255,255,255,0.22);
            }
            QPushButton#btn-end { background:#FF4D4F; color:#FFF; }
            QPushButton#btn-end:hover { background:#FF6B6D; }
        """)
        lay = QHBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(12)
        self._btn_info = QPushButton("내 정보", self); self._btn_info.setObjectName("btn-info")
        self._btn_end = QPushButton("운동 종료", self); self._btn_end.setObjectName("btn-end")
        lay.addWidget(self._btn_info); lay.addWidget(self._btn_end)
        self._btn_end.clicked.connect(self.endClicked.emit)
        self._btn_info.clicked.connect(self.infoClicked.emit)
    def set_enabled(self, end_enabled=True, info_enabled=True):
        self._btn_end.setEnabled(end_enabled); self._btn_info.setEnabled(info_enabled)

# ──────────────────────────────────────────────
# 🎯 관절각도 패널
# ──────────────────────────────────────────────
class PoseAnglePanel(QWidget):
    ROW_KEYS = [
        ("Knee","Knee(L)","Knee(R)"),
        ("Hip","Hip(L)","Hip(R)"),
        ("Shoulder","Shoulder(L)","Shoulder(R)"),
        ("Elbow","Elbow(L)","Elbow(R)"),
        ("HipLine","HipLine(L)","HipLine(R)"),
    ]
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget { background: rgba(0,0,0,160); border-radius:16px; }
            QLabel#hdr { color:#BFC6CF; font-size:16px; font-weight:700; }
            QLabel#cell { color:#FFFFFF; font-size:16px; font-weight:600; }
            QLabel#cell-dim { color:#BFC6CF; font-size:14px; font-weight:600; }
            QLabel#title { color:#FFFFFF; font-size:18px; font-weight:800; }
        """)
        wrap = QVBoxLayout(self); wrap.setContentsMargins(14,12,14,12); wrap.setSpacing(8)
        title = QLabel("Pose Angles (°)", self); title.setObjectName("title"); wrap.addWidget(title)
        grid_host = QWidget(self); grid = QGridLayout(grid_host)
        grid.setContentsMargins(0,0,0,0); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(6)
        hdrs = ["Part","L","R","Avg"]
        for c,h in enumerate(hdrs):
            lab = QLabel(h,self); lab.setObjectName("hdr"); grid.addWidget(lab,0,c)
        self._cells: dict[str,QLabel] = {}
        for r,(part,lkey,rkey) in enumerate(self.ROW_KEYS,start=1):
            lab_part = QLabel(part,self); lab_part.setObjectName("cell-dim"); grid.addWidget(lab_part,r,0)
            for c,col in enumerate(("L","R","Avg"),start=1):
                lab = QLabel("-",self); lab.setObjectName("cell"); grid.addWidget(lab,r,c)
                self._cells[f"{part}:{col}"] = lab
        wrap.addWidget(grid_host)
    @staticmethod
    def _fmt(v: Optional[float]) -> str:
        if v is None: return "-"
        try: return f"{float(v):.1f}"
        except: return "-"
    def set_angles(self, angles: dict):
        for part,lkey,rkey in self.ROW_KEYS:
            l = angles.get(lkey); r = angles.get(rkey)
            avg = None
            if isinstance(l,(int,float)) and isinstance(r,(int,float)):
                avg = (float(l)+float(r))/2.0
            self._cells[f"{part}:L"].setText(self._fmt(l))
            self._cells[f"{part}:R"].setText(self._fmt(r))
            self._cells[f"{part}:Avg"].setText(self._fmt(avg))

# ──────────────────────────────────────────────
# 🧩 추가: 실시간 AI 출력 모니터 패널
# ──────────────────────────────────────────────
class DualAIMonitor(QWidget):
    """
    좌측 하단: 실시간 FI/BI/Tempo 모니터
    (reps_pred_dual.tsv + imu_tempo.tsv)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget { background: rgba(0,0,0,160); border-radius:16px; }
            QLabel { color:#FFFFFF; font-size:16px; font-weight:600; background:transparent; }
            QLabel#title { font-size:18px; font-weight:800; color:#00E0FF; }
        """)
        lay = QVBoxLayout(self); lay.setContentsMargins(16,12,16,12); lay.setSpacing(6)
        title = QLabel("AI 추론 결과",self); title.setObjectName("title")
        self.lbl_fi = QLabel("FI_L: -   FI_R: -   BI: -")
        self.lbl_tempo = QLabel("TempoCV: -   Level: -")
        self.lbl_stage = QLabel("Stage: -")
        lay.addWidget(title); lay.addWidget(self.lbl_fi); lay.addWidget(self.lbl_tempo); lay.addWidget(self.lbl_stage)
        lay.addStretch(1)

    def update_values(self, fi_l, fi_r, bi, tempo, level, stage_text=""):
        self.lbl_fi.setText(f"FI_L: {fi_l:.2f}   FI_R: {fi_r:.2f}   BI: {bi:.2f}")
        self.lbl_tempo.setText(f"TempoCV: {tempo:.2f}   Level: {level}")
        self.lbl_stage.setText(f"Stage: {stage_text or '-'}")
