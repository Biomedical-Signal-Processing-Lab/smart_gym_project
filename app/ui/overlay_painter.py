# SQUAT/app/ui/overlay_painter.py
from dataclasses import dataclass, field
from typing import Tuple, Optional
from PySide6.QtGui import QPainter, QFont, QColor, QPainterPath, QPen, QImage
from PySide6.QtCore import Qt

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
        pen = QPen(style.outline_color, style.outline_width)
        painter.setPen(pen)
        painter.drawPath(path)

    painter.setPen(Qt.NoPen)
    painter.fillPath(path, style.color)
    painter.end()

def draw_count_top_left(img: QImage, count: int, style: Optional[TextStyle] = None):
    if style is None:
        base_px = max(28, img.width() // 18)
        style = TextStyle(px=base_px, margin=int(base_px*0.6))

    x = style.margin
    y = style.margin + style.px   # addText는 베이스라인 기준이므로 px만큼 내려서
    draw_text(img, f"COUNT {count}", x, y, style)
