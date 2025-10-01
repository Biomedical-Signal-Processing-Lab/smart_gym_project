# ui/overlay_painter.py
from dataclasses import dataclass, field
from typing import Optional
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPainterPath, QPen
from PySide6.QtCore import Qt, QSize, QRect

# ===== 영상 + 오버레이 컨테이너 =====
class VideoCanvas(QWidget):
    _anchor_to_cell = {
        "top-left": (0,0), "top-center": (0,1), "top-right": (0,2),
        "center-left": (1,0), "center": (1,1), "center-right": (1,2),
        "bottom-left": (2,0), "bottom-center": (2,1), "bottom-right": (2,2),
    }

    def __init__(self, min_size: QSize | None = None, parent=None):
        super().__init__(parent)
        # 라벨(영상 표시)
        self._video = QLabel("camera: no frame", self)
        self._video.setAlignment(Qt.AlignCenter)
        if min_size:
            self._video.setMinimumSize(min_size)

        # 오버레이 루트(투명)
        self._overlay_root = QWidget(self)
        self._overlay_root.setAttribute(Qt.WA_StyledBackground, False)
        self._overlay_root.setStyleSheet("background: transparent;")

        # 3x3 그리드(앵커 셀)
        grid = QGridLayout(self._overlay_root)
        grid.setContentsMargins(12,12,12,12)
        grid.setSpacing(8)
        self._cells = [[QWidget(self._overlay_root) for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                v_align = Qt.AlignTop if r == 0 else (Qt.AlignVCenter if r == 1 else Qt.AlignBottom)
                grid.addWidget(self._cells[r][c], r, c, alignment=v_align)

        # 마지막 프레임과 원본 해상도 기록
        self._last_qimage: QImage | None = None
        self._img_w = None
        self._img_h = None

    # ------ 내부: 현재 위젯 크기에서 영상 표시 rect 계산 ------
    def _compute_target_rect(self) -> QRect:
        W, H = self.width(), self.height()
        if not self._img_w or not self._img_h or W <= 0 or H <= 0:
            return QRect(0, 0, W, H)

        sf = min(W / self._img_w, H / self._img_h)
        disp_w = int(self._img_w * sf)
        disp_h = int(self._img_h * sf)
        x = (W - disp_w) // 2
        y = (H - disp_h) // 2
        return QRect(x, y, disp_w, disp_h)

    # ------ 내부: 라벨/오버레이 포지션 업데이트 ------
    def _position_layers(self):
        rect = self._compute_target_rect()
        # 라벨/오버레이를 동일 rect로 맞춰 정확히 겹치게
        self._video.setGeometry(rect)
        self._overlay_root.setGeometry(rect)

        # 라벨 픽스맵도 rect 크기에 맞춰 스케일
        if self._last_qimage is not None and rect.width() > 0 and rect.height() > 0:
            pm = QPixmap.fromImage(self._last_qimage).scaled(
                rect.width(), rect.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._video.setPixmap(pm)

    # ------ 외부 API: 프레임 설정 ------
    def set_frame(self, qimage: QImage):
        """마지막 프레임을 저장하고, 라벨/오버레이를 영상 표시 영역에 딱 맞춰 배치"""
        self._last_qimage = qimage
        self._img_w = qimage.width()
        self._img_h = qimage.height()
        self._position_layers()

    # ------ 오버레이 추가 ------
    def add_overlay(self, widget: QWidget, anchor: str = "top-right"):
        """오버레이 위젯을 지정 위치 셀의 레이아웃에 추가(가로 정렬 유지)"""
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
