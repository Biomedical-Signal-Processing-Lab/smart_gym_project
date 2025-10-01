# ui/topbar.py
from __future__ import annotations
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QSizePolicy,
    QGraphicsDropShadowEffect, QFrame
)
from pathlib import Path

class TopBar(QWidget):
    logoClicked = Signal()
    backClicked = Signal()

    def __init__(self, parent: Optional[QWidget]=None, *, show_back: bool = False,
                 show_info_button: bool = True, show_logout_button: bool = True):
        super().__init__(parent)

        self.setObjectName("TopBar")
        self.setFixedHeight(80)
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.setStyleSheet("""
            #TopBar {
                background: black;   
            }

            QLabel#CenterTitle {
                color: #475569;        
                font-size: 15px;
                font-weight: 700;
            }

            QPushButton#NavBtn {
                border: none;
                background: transparent;
                padding: 8px 10px;
                color: #334155;        
            }
            QPushButton#NavBtn:hover {
                background: rgba(0,0,0,0.05);
                border-radius: 6px;
            }

            QPushButton#InfoBtn {
                color: white;
                font-weight: 600;
            }
            QPushButton#InfoBtn:hover {
                background: rgba(37,99,235,0.08);
                border-radius: 6px;
            }

            QPushButton#LogoutBtn {
                color: white;
                font-weight: 600;
            }
            QPushButton#LogoutBtn:hover {
                background: rgba(220,38,38,0.08);
                border-radius: 6px;
            }

            QFrame#BottomLine {
                background: rgba(0,0,0,0.08);
            }
        """)

        left = QHBoxLayout(); left.setSpacing(8); left.setContentsMargins(0,0,0,0)

        self.btn_back = QPushButton("←")
        self.btn_back.setObjectName("NavBtn")
        self.btn_back.setVisible(show_back)
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self._on_back)

        self.logo = QLabel()  
        self.logo.setObjectName("LogoLabel")
        self.logo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.logo.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.logo.setCursor(Qt.PointingHandCursor)
        self.logo.mousePressEvent = lambda e: self._on_logo()

        left.addWidget(self.btn_back)
        left.addWidget(self.logo)

        app_root = Path(__file__).resolve().parent.parent  
        logo_path = app_root / "assets" / "healthking.png"
        self._logo_src = QPixmap(str(logo_path))

        self.centerTitle = QLabel("")
        self.centerTitle.setObjectName("CenterTitle")
        self.centerTitle.setAlignment(Qt.AlignCenter)
        self.centerTitle.setVisible(False)

        self.btn_info: Optional[QPushButton] = None
        self.btn_logout: Optional[QPushButton] = None

        right = QHBoxLayout(); right.setSpacing(6); right.setContentsMargins(0,0,0,0)

        if show_info_button:
            self.btn_info = QPushButton("내 정보")
            self.btn_info.setObjectName("InfoBtn")
            self.btn_info.setCursor(Qt.PointingHandCursor)
            self.btn_info.clicked.connect(self._goto_info)
            right.addWidget(self.btn_info)

        if show_logout_button:
            self.btn_logout = QPushButton("로그아웃")
            self.btn_logout.setObjectName("LogoutBtn")
            self.btn_logout.setCursor(Qt.PointingHandCursor)
            self.btn_logout.clicked.connect(self._logout)
            right.addWidget(self.btn_logout)

        h = QHBoxLayout(self)
        h.setContentsMargins(16, 6, 16, 6)
        h.setSpacing(8)

        leftWrap = QWidget(); leftWrap.setLayout(left)
        rightWrap = QWidget(); rightWrap.setLayout(right)

        h.addWidget(leftWrap, 0, Qt.AlignVCenter)
        h.addStretch(1)
        h.addWidget(self.centerTitle, 0, Qt.AlignCenter)
        h.addStretch(1)
        h.addWidget(rightWrap, 0, Qt.AlignRight | Qt.AlignVCenter)

        line = QFrame(self)
        line.setObjectName("BottomLine")
        line.setFixedHeight(1)
        line.setGeometry(0, self.height()-1, self.width(), 1)
        line.lower()
        self._bottomLine = line

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 2)
        shadow.setColor(Qt.black)
        self.setGraphicsEffect(shadow)

        self._update_logo_pixmap()

    def _update_logo_pixmap(self):
        if self._logo_src.isNull():
            self.logo.setText("헬스왕")
            self.logo.setStyleSheet("color:#1E293B; font-size:18px; font-weight:900; letter-spacing:0.5px;")
            return

        max_h = max(1, self.height() - 16)  
        self.logo.setPixmap(self._logo_src.scaledToHeight(max_h, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._bottomLine.setGeometry(0, self.height()-1, self.width(), 1)
        self._update_logo_pixmap()

    def _find_router(self):
        w = self.parent()
        while w and not hasattr(w, "navigate"):
            w = w.parent()
        return w

    def _get_ctx(self):
        w = self.parent()
        while w and not hasattr(w, "ctx"):
            w = w.parent()
        return getattr(w, "ctx", None)

    def _navigate(self, page: str):
        router = self._find_router()
        if router:
            router.navigate(page)

    def _on_logo(self):
        self.logoClicked.emit()
        self._navigate("start")

    def _on_back(self):
        self.backClicked.emit()
        self._navigate("start")

    def _goto_info(self):
        self._navigate("info")

    def _logout(self):
        ctx = self._get_ctx()
        if ctx and getattr(ctx, "clear_current_user", None):
            try:
                ctx.clear_current_user()
            except Exception:
                pass
        self._navigate("start")

    def setTitle(self, text: str, visible: bool=True):
        self.centerTitle.setText(text)
        self.centerTitle.setVisible(bool(visible and text))

    def setBackVisible(self, visible: bool):
        self.btn_back.setVisible(visible)

    def hideInfoButton(self):
        if self.btn_info: self.btn_info.hide()

    def hideLogoutButton(self):
        if self.btn_logout: self.btn_logout.hide()
