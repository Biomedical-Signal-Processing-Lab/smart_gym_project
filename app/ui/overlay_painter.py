# ui/overlay_painter.py
from dataclasses import dataclass, field
from typing import Optional
from PySide6.QtCore import Qt, QSize, QRect, Signal, QObject
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QHBoxLayout, QPushButton
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPainterPath, QPen

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

class CanvasHUD(QObject):
    endClicked = Signal()

    def __init__(self, canvas: VideoCanvas, *, count_label_text: str = "SQUAT"):
        super().__init__(canvas)
        self.canvas = canvas
        self._count_label_text = count_label_text

        self._lbl_count = QLabel(f"{self._count_label_text}: 0", canvas)
        self._lbl_count.setStyleSheet("""
            QLabel {
                background: rgba(0,0,0,120);
                color: white;
                padding: 6px 10px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 18px;
            }
        """)
        self._btn_end = QPushButton("운동 종료", canvas)
        self._btn_end.setStyleSheet(
            "background: rgba(0,0,0,120); color: white; border-radius: 10px; padding: 6px 12px;"
        )
        self._btn_end.clicked.connect(self.endClicked.emit)
        self.mount()

    # --- API ---
    def mount(self):
        self.canvas.add_overlay(self._lbl_count, anchor="top-left")
        self.canvas.add_overlay(self._btn_end, anchor="top-right")
        self._lbl_count.show()
        self._btn_end.show()

    def set_count(self, n: int):
        self._lbl_count.setText(f"{self._count_label_text}: {n}")

    def set_count_visible(self, visible: bool):
        if self._lbl_count:
            self._lbl_count.setVisible(visible)

    def set_end_visible(self, visible: bool):
        if self._btn_end:
            self._btn_end.setVisible(visible)

    def set_end_enabled(self, enabled: bool):
        self._btn_end.setEnabled(enabled)

    def teardown(self):
        self._lbl_count.hide()
        self._btn_end.hide()

@dataclass
class TextStyle:
    family: str = ""
    bold: bool = True
    px: int = 36
    color: QColor = field(default_factory=lambda: QColor(255, 255, 255))
    outline_color: QColor = field(default_factory=lambda: QColor(0, 0, 0))
    outline_width: int = 3
    margin: int = 16

def draw_text(img: QImage, text: str, x: int, y: int, style: TextStyle):
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)

    font = QFont(style.family)
    font.setBold(style.bold)
    font.setPixelSize(style.px)
    painter.setFont(font)

    path = QPainterPath()
    path.addText(x, y, font, text)

    if style.outline_width > 0:
        painter.setPen(QPen(style.outline_color, style.outline_width))
        painter.drawPath(path)

    painter.setPen(Qt.NoPen)
    painter.fillPath(path, style.color)
    painter.end()

def draw_count_top_left(img: QImage, count: int, label: str, style: Optional[TextStyle] = None):
    if style is None:
        base_px = max(28, img.width() // 18)
        style = TextStyle(px=base_px, margin=int(base_px * 0.6))
    x = style.margin
    y = style.margin + style.px
    draw_text(img, f"{label}: {count}", x, y, style)
