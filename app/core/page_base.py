# core/page_base.py
from PySide6.QtWidgets import QWidget

class PageBase(QWidget):
    def on_enter(self, ctx):  # ctx: AppContext
        pass
    def on_leave(self, ctx):
        pass
