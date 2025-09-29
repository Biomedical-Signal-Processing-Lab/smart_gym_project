# app.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtCore import Qt
from core.context import AppContext
from core.router import Router
from views.start_page import StartPage
from views.squat_page import SquatPage
from views.select_page import WorkoutSelectPage

class MainWindow(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("운동 분석")
        self.router = Router(ctx, parent=self)
        self.setCentralWidget(self.router)

        # 페이지 등록
        self.router.register("start", lambda: StartPage())
        self.router.register("select", lambda: WorkoutSelectPage())
        self.router.register("squat", lambda: SquatPage())

        # 단축키
        QShortcut(QKeySequence(Qt.Key_F11), self).activated.connect(self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_Escape), self).activated.connect(self.showNormal)
        QShortcut(QKeySequence(Qt.Key_F1), self).activated.connect(lambda: self.router.navigate("start"))
        QShortcut(QKeySequence(Qt.Key_F2), self).activated.connect(lambda: self.router.navigate("squat"))

        self.resize(800, 800)
        self.router.navigate("start")

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ctx = AppContext()
    win = MainWindow(ctx)
    win.show()
    sys.exit(app.exec())
