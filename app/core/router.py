# core/router.py
from PySide6.QtWidgets import QStackedWidget
from typing import Dict, Callable
from core.page_base import PageBase

# 페이지 등록/전환 관리
class Router(QStackedWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._factories: Dict[str, Callable[[], PageBase]] = {}
        self._pages: Dict[str, PageBase] = {}

    def register(self, name: str, factory: Callable[[], PageBase]):
        self._factories[name] = factory

    def navigate(self, name: str):
        cur = self.currentWidget()
        if isinstance(cur, PageBase):
            cur.on_leave(self.ctx)

        if name not in self._pages:
            page = self._factories[name]()
            self._pages[name] = page
            self.addWidget(page)

        page = self._pages[name]
        self.setCurrentWidget(page)
        # 새 페이지 on_enter 호출
        if isinstance(page, PageBase):
            page.on_enter(self.ctx)
