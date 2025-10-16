# main.py
import os
import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtCore import Qt, QLocale
from core.context import AppContext
from core.router import Router

from views.start_page import StartPage
from views.exercise_page import ExercisePage
from views.guide_page import GuidePage
from views.summary_page import SummaryPage
from views.enroll_page import EnrollPage
from views.info_page import InfoPage

class MainWindow(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("자세어때")
        self.router = Router(ctx, parent=self)
        self.setCentralWidget(self.router)
        self.ctx.set_router(self.router)

        self.router.register("start", lambda: StartPage())
        self.router.register("guide", lambda: GuidePage())
        self.router.register("exercise", lambda: ExercisePage())
        self.router.register("summary", lambda: SummaryPage())
        self.router.register("enroll", lambda: EnrollPage())
        self.router.register("info",   lambda: InfoPage())

        QShortcut(QKeySequence(Qt.Key_F11), self).activated.connect(self._toggle_fullscreen)
        QShortcut(QKeySequence(Qt.Key_F1), self).activated.connect(lambda: self.router.navigate("start"))
        QShortcut(QKeySequence(Qt.Key_F2), self).activated.connect(lambda: self.router.navigate("guide"))
        QShortcut(QKeySequence(Qt.Key_F3), self).activated.connect(lambda: self.router.navigate("exercise"))
        QShortcut(QKeySequence(Qt.Key_F4), self).activated.connect(lambda: self.router.navigate("info"))

        self.resize(1280, 800)
        self.router.navigate("start")

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

if __name__ == "__main__":
    # os.environ["QT_IM_MODULE"] = "qtvirtualkeyboard"
    QLocale.setDefault(QLocale(QLocale.Korean, QLocale.SouthKorea))
    
    app = QApplication(sys.argv)
    ctx = AppContext()
    win = MainWindow(ctx)
    win.show()
    sys.exit(app.exec())
