# SQUAT/app/ui/score_painter.py
from PySide6.QtWidgets import QWidget, QLabel, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QPoint, QPropertyAnimation, QEasingCurve, QVariantAnimation
from PySide6.QtGui import QColor

class ScoreOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

    def resizeEvent(self, e):
        self.setGeometry(self.parent().rect())
        return super().resizeEvent(e)

    def show_score(self, text: str = "100", base_px: int | None = None,
                   text_qcolor: QColor | None = None):
        lbl = QLabel(text, self)
        lbl.setAttribute(Qt.WA_TranslucentBackground, True)
        lbl.setAlignment(Qt.AlignCenter)
        px = base_px if base_px else max(100, int(self.width() * 0.08))
        tc = text_qcolor or QColor(0, 128, 255)
        lbl.setStyleSheet(f"""
            QLabel {{
                color: rgba({tc.red()},{tc.green()},{tc.blue()},{tc.alpha()});
                font: 1000 {px}px "Pretendard";
                padding: 8px 16px;
                background: rgba(0,0,0,70);        
                border-radius: 12px;
                border: 2px solid rgba(0,0,0,120); 
            }}
        """)
        lbl.adjustSize()

        # 중앙 위치에 배치
        x = (self.width() - lbl.width()) // 2
        y = (self.height() - lbl.height()) // 2
        lbl.move(x, y)
        lbl.show()

        # 투명도 효과
        fx = QGraphicsOpacityEffect(lbl)
        lbl.setGraphicsEffect(fx)
        fx.setOpacity(0.0) 

        # 1) 팝(등장) 연출: 0.0→1.0로 페이드인 + 가벼운 '커짐' 효과
        fade_in = QPropertyAnimation(fx, b"opacity", self)
        fade_in.setDuration(120)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.OutCubic)

        # 폰트 사이즈 살짝 0.85x → 1.0x로 키우기 (QVariantAnimation로 간단 스케일)
        start_px = int(px * 0.85)
        pop_anim = QVariantAnimation(self)
        pop_anim.setDuration(120)
        pop_anim.setStartValue(start_px)
        pop_anim.setEndValue(px)
        pop_anim.setEasingCurve(QEasingCurve.OutBack)

        def _apply_font(v):
            lbl.setStyleSheet(f"""
                QLabel {{
                    color: rgba({tc.red()},{tc.green()},{tc.blue()},{tc.alpha()});
                    font: 1000 {int(v)}px "Pretendard";
                    padding: 8px 16px;
                    background: rgba(0,0,0,70);
                    border-radius: 12px;
                    border: 2px solid rgba(0,0,0,120);
                }}
            """)
            # 크기 바뀌면 가운데 유지
            old = lbl.pos()
            lbl.adjustSize()
            lbl.move(old.x() - (lbl.width() - lbl.sizeHint().width())//2,
                     old.y() - (lbl.height()- lbl.sizeHint().height())//2)
        pop_anim.valueChanged.connect(_apply_font)

        # 2) 위로 떠오르며 사라짐: 위치 y-80, 불투명도 1→0
        move = QPropertyAnimation(lbl, b"pos", self)
        move.setDuration(900)
        move.setStartValue(QPoint(x, y))
        move.setEndValue(QPoint(x, y - 80))
        move.setEasingCurve(QEasingCurve.OutCubic)

        fade_out = QPropertyAnimation(fx, b"opacity", self)
        fade_out.setDuration(900)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.InQuad)

        # 정리
        def cleanup():
            lbl.deleteLater()
        fade_out.finished.connect(cleanup)

        # 시퀀스 실행: 팝(120ms) → 떠오르며 페이드(900ms)
        fade_in.start()
        pop_anim.start()
        move.start()
        fade_out.start()
