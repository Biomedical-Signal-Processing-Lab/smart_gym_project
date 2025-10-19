# guide_page.py  
from typing import List, Optional, Iterable
from PySide6.QtCore import Qt, QSize, QUrl
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFrame, QSizePolicy, QSpacerItem
)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from core.page_base import PageBase
from data.guide_data import Exercise, list_all
from ui.guide_style import (
    style_page_root, style_side_panel, style_scrollarea, style_exercise_card, 
    style_info_card, style_header_title, style_header_chip, style_header_desc, force_bg
)
import os

# ---------------- UI Util (배경 자원 경로) ----------------
def asset_path(*parts) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    return os.path.join(root, *parts)

def _clear_layout(layout) -> None:
    while layout and layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        c = item.layout()
        if w is not None:
            w.deleteLater()
        elif c is not None:
            _clear_layout(c)

class ExerciseCard(QFrame):
    def __init__(self, info: Exercise, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.info = info
        self.setObjectName("ExerciseCard")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(6)

        title_line = QHBoxLayout()
        title_line.setContentsMargins(0, 0, 0, 0)
        title_line.setSpacing(8)

        title = QLabel(info.title)
        title.setObjectName("title")
        title.setStyleSheet("background:transparent;")
        title.setWordWrap(False)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        cat = QLabel(info.category)
        cat.setObjectName("chip")
        cat.setStyleSheet("background:transparent;")

        title_line.addWidget(title, 1)
        title_line.addWidget(cat, 0, Qt.AlignVCenter)

        sets = QLabel(info.sets_reps)
        sets.setObjectName("sets")
        sets.setStyleSheet("background:transparent;")

        root.addLayout(title_line)

        force_bg(self, """
            QFrame#ExerciseCard {
                background:#ffffff;
                border:1px solid rgba(0,0,0,0.06);
                border-radius:16px;
                min-height:140px;                 
                padding:18px 24px;              
            }
            QFrame#ExerciseCard[selected="true"] {
                background:#1a73e8;
                border:1px solid #1a73e8;
            }
            QFrame#ExerciseCard[selected="true"] QLabel#title,
            QFrame#ExerciseCard[selected="true"] QLabel#sets,
            QFrame#ExerciseCard[selected="true"] QLabel#chip { color:white; }
            QLabel#title { font-size:50px; font-weight:700; color:#0f172a; }
            QLabel#sets  { color:#6b7380; font-size:16px; }
            QLabel#chip  {
                background:#eef4ff; border:1px solid rgba(25,118,210,0.5); border-radius:999px;
                padding:2px 10px; color:#24527a; font-size:16px; font-weight:600;
            }
        """)

    def setSelected(self, v: bool) -> None:
        self.setProperty("selected", v)
        self.style().unpolish(self)
        self.style().polish(self)

class InfoCard(QFrame):
    def __init__(self, title: str, body_widget: QWidget):
        super().__init__()
        self.setObjectName("InfoCard")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 22, 22, 22)
        lay.setSpacing(12)

        ttl = QLabel(title)
        ttl.setObjectName("CardTitle")
        ttl.setStyleSheet("background:transparent; color:#111827; font-weight:800; font-size:36px;")
        lay.addWidget(ttl)

        def paint_texts(w: QWidget):
            if isinstance(w, QLabel):
                w.setStyleSheet("background:transparent; color:#0f172a; font-size:24px;")
            for ch in w.findChildren(QLabel):
                ch.setStyleSheet("background:transparent; color:#0f172a; font-size:24px;")
        paint_texts(body_widget)

        lay.addWidget(body_widget)

        force_bg(self, """
            #InfoCard {
                background: rgba(255,255,255,0.96);
                border: 1px solid rgba(0,0,0,0.05);
                border-radius: 18px;
            }
        """)

def bullet_list(items: Iterable[str], numbered: bool = False) -> QWidget:
    w = QWidget()
    force_bg(w, "background:transparent;")
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(8)
    for i, t in enumerate(items, 1):
        line = QHBoxLayout()
        line.setSpacing(10)
        dot = QLabel(str(i) if numbered else "•")
        dot.setStyleSheet("background:transparent; color:#4b5563; font-size:24px;")
        dot.setFixedWidth(24)
        lbl = QLabel(t)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("""
            background:rgba(25,118,210,0.10);
            color:#0f172a; font-size:24px; font-weight:500; border-radius:12px; padding:12px 14px;
        """)
        line.addWidget(dot)
        line.addWidget(lbl, 1)
        v.addLayout(line)
    return w

class IconButton(QPushButton):
    def __init__(self, png_path: str, size: Optional[int | QSize | tuple[int, int]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._base_pix = QPixmap(png_path)
        self._target_size: Optional[QSize] = None

        if size is not None:
            if isinstance(size, int):
                self._target_size = QSize(size, size)
            elif isinstance(size, tuple):
                self._target_size = QSize(size[0], size[1])
            else:
                self._target_size = size

        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("border:0; background:transparent; padding:0; margin:0;")
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._update_icon()

    def _scaled_pix(self) -> QPixmap:
        if self._target_size:
            return self._base_pix.scaled(self._target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return self._base_pix

    def _update_icon(self) -> None:
        pix = self._scaled_pix()
        self.setIcon(QIcon(pix))
        self.setIconSize(pix.size())
        self.setFixedSize(pix.size())

    def setPixmap(self, pix: QPixmap, size: Optional[int | QSize | tuple[int, int]] = None) -> None:
        self._base_pix = pix
        if size is not None:
            if isinstance(size, int):
                self._target_size = QSize(size, size)
            elif isinstance(size, tuple):
                self._target_size = QSize(size[0], size[1])
            else:
                self._target_size = size
        self._update_icon()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._update_icon()

class GuidePage(PageBase):
    ICON_SIZE = 200

    def __init__(self):
        super().__init__()
        self.setObjectName("GuidePage")

        # ===== 배경 + 글래스 패널 래퍼 추가 (UI만) =====
        self._bg_pix = None
        self._bg = QLabel(self)             
        self._bg.setScaledContents(False)
        self._bg.lower()

        bg_path = asset_path("assets", "background", "bg_gym.jpg")
        if os.path.exists(bg_path):
            pm = QPixmap(bg_path)
            if not pm.isNull():
                self._bg_pix = pm
                self._rescale_bg()

        # 글래스 패널 (기존 root 레이아웃을 이 안에 구성)
        self._panel = QFrame(self)
        self._panel.setObjectName("glassPanel")
        self._panel.setAttribute(Qt.WA_StyledBackground, True)

        style_page_root(self)  # 기존 호출은 유지 (테마 변수 사용 중이면 활용됨)

        self.exercises = list_all()  # 로직 유지

        # === 패널 내부 루트 레이아웃 ===
        panel_root = QVBoxLayout(self._panel)
        panel_root.setContentsMargins(28, 28, 28, 28)
        panel_root.setSpacing(18)

        # ---------- TopBar (새 UI) ----------
        top = QFrame()
        top.setObjectName("TopBar")
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(22, 18, 22, 18)
        top_lay.setSpacing(14)

        self.btn_back = QPushButton("←")
        self.btn_back.setObjectName("BtnBack")
        self.btn_back.setFixedHeight(72)
        self.btn_back.clicked.connect(lambda: self._goto("start"))

        self.title = QLabel("운동 가이드")
        self.title.setObjectName("Title")
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        top_lay.addWidget(self.btn_back)
        top_lay.addSpacing(12)
        top_lay.addWidget(self.title, 1)

        self.btn_user = QPushButton("로그인 필요")
        self.btn_user.setObjectName("BtnUser")
        self.btn_user.setFixedHeight(68)
        self.btn_user.clicked.connect(lambda: self._goto("info"))

        self.btn_top_start = QPushButton("▶  운동 시작")
        self.btn_top_start.setObjectName("BtnStart")
        self.btn_top_start.setFixedHeight(68)
        self.btn_top_start.clicked.connect(lambda: self._goto("exercise"))

        top_lay.addWidget(self.btn_user)
        top_lay.addWidget(self.btn_top_start)

        # ---------- 본문 영역 ----------
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(18)

        # 좌측 패널 (리스트)
        self.left_panel = self._build_left_panel()
        body.addWidget(self.left_panel, 2)

        # 우측 디테일 패널 (동영상 + 주의사항만)
        self.detail_panel = self._build_detail_panel()
        body.addWidget(self.detail_panel, 3)

        panel_root.addWidget(top)
        panel_root.addLayout(body, 1)

        # 스타일시트 적용
        self.setStyleSheet(self._stylesheet())

        if self.exercises:
            self._select(self.exercises[0])

    # ---------- 기존 좌측 패널 로직/바인딩 유지, 스타일만 변경 ----------
    def _build_left_panel(self) -> QWidget:
        side = QFrame()
        side.setObjectName("LeftMenu")
        style_side_panel(side)  

        side.setFixedWidth(440)

        v = QVBoxLayout(side)
        v.setContentsMargins(14, 14, 14, 14)
        v.setSpacing(12)

        scroll = QScrollArea()
        style_scrollarea(scroll)

        content = QWidget()
        force_bg(content, "background:transparent;")
        lv = QVBoxLayout(content)
        lv.setContentsMargins(8, 8, 4, 8)
        lv.setSpacing(12)

        self._cards: List[ExerciseCard] = []
        for ex in self.exercises:
            card = ExerciseCard(ex)
            style_exercise_card(card)  
            card.mousePressEvent = lambda e, _ex=ex: self._select(_ex)
            self._cards.append(card)
            lv.addWidget(card)

        lv.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        v.addWidget(scroll, 1)
        return side

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        force_bg(panel, "background:transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self.h_title = QLabel("")
        self.h_title.setObjectName("VideoTitle")
        style_header_title(self.h_title)

        self.h_cate = QLabel("")
        style_header_chip(self.h_cate)

        self.h_desc = QLabel("")
        self.h_desc.setTextFormat(Qt.PlainText)
        self.h_desc.setWordWrap(True)
        style_header_desc(self.h_desc)

        self.v_goal = QLabel("")
        self.v_goal.setWordWrap(True)
        self.v_reco = QLabel("")
        self.v_reco.setWordWrap(True)
        self.v_cate = QLabel("")
        self.v_cate.setWordWrap(True)

        # ------- 동영상 카드 -------
        self.video_card = QFrame()
        self.video_card.setObjectName("VideoCard")
        video_box = QVBoxLayout(self.video_card)
        video_box.setContentsMargins(22, 22, 22, 22)
        video_box.setSpacing(8)

        video_title = QLabel("동영상")
        video_title.setObjectName("VideoTitle")

        self._video_widget = QVideoWidget()
        self._video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_box.addWidget(video_title)
        video_box.addWidget(self._video_widget, 1)

        # 플레이어
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(0.0)  # 무음
        self._player.setAudioOutput(self._audio_output)
        self._player.setVideoOutput(self._video_widget)

        demo_video = asset_path("assets", "videos", "jm_guide.mov")
        self._set_player_source(demo_video)
        self._player.mediaStatusChanged.connect(self._loop_video)
        self._player.play()

        # ------- 주의사항 카드 -------
        tips_card = QFrame()
        tips_card.setObjectName("TipsCard")
        tips_box = QVBoxLayout(tips_card)
        tips_box.setContentsMargins(22, 22, 22, 22)
        tips_box.setSpacing(12)

        self.tips_title = QLabel("주의사항 및 팁")
        self.tips_title.setObjectName("TipsTitle")
        self.tips_widget = bullet_list([], numbered=False)

        tips_box.addWidget(self.tips_title)
        tips_box.addWidget(self.tips_widget)

        self.steps_widget = bullet_list([], numbered=True)

        layout.addWidget(self.video_card, 1)
        layout.addWidget(tips_card, 1)

        return panel

    def _select(self, ex: Exercise) -> None:
        for c in self._cards:
            c.setSelected(c.info.key == ex.key)
        self.h_title.setText(ex.title)
        self.h_cate.setText(ex.category)
        self.h_desc.setText(ex.description)
        self.v_goal.setText(ex.goal_muscles)
        self.v_reco.setText(ex.recommend)
        self.v_cate.setText(ex.category)
        self._replace_bullet(self.steps_widget, ex.steps, True)
        self._replace_bullet(self.tips_widget, ex.tips, False)

    def _replace_bullet(self, container: QWidget, items: Iterable[str], numbered: bool = False) -> None:
        lay = container.layout()
        _clear_layout(lay)
        for i, t in enumerate(items, 1):
            row = QHBoxLayout()
            row.setSpacing(10)
            dot = QLabel(str(i) if numbered else "•")
            dot.setFixedWidth(24)
            dot.setStyleSheet("background:transparent; color:#4b5563; font-size:24px;")
            lbl = QLabel(t)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("""
                background:rgba(25,118,210,0.10); color:#0f172a; font-size:24px; font-weight:500;
                border-radius:12px; padding:12px 14px;
            """)
            row.addWidget(dot)
            row.addWidget(lbl, 1)
            lay.addLayout(row)

    def _goto(self, page: str) -> None:
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)

    def on_enter(self, ctx):
        try:
            if getattr(ctx, "current_user_name", None):
                self.btn_user.setText(f"{ctx.current_user_name} 님")
            else:
                self.btn_user.setText("홍길동 님")
        except Exception:
            self.btn_user.setText("홍길동 님")

    # ===== 배경/패널/비디오 레이아웃 처리 (UI만) =====
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale_bg()
        self._layout_panel()
        self._fix_video_ratio()

    def _layout_panel(self):
        w, h = self.width(), self.height()
        target_w = max(int(w * 0.95), 1100)
        target_h = max(int(h * 0.95), 720)
        x = (w - target_w) // 2
        y = (h - target_h) // 2
        self._panel.setGeometry(x, y, target_w, target_h)

    def _rescale_bg(self):
        if self._bg_pix:
            self._bg.setGeometry(self.rect())
            scaled = self._bg_pix.scaled(
                self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            self._bg.setPixmap(scaled)

    def _fix_video_ratio(self):
        if not self.video_card:
            return
        avail_w = max(self.video_card.width() - 44, 1)
        by_width_h = int(avail_w * 9 / 16)

        right_h = max(self.video_card.parentWidget().height(), 1)
        max_h = int(right_h * 0.55)

        target_h = min(by_width_h, max_h)
        target_w = int(target_h * 16 / 9)
        self._video_widget.setFixedSize(QSize(target_w, target_h))

    # ===== 비디오 유틸 =====
    def _loop_video(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self._player.setPosition(0)
            self._player.play()

    def _set_player_source(self, path: str):
        if not os.path.isabs(path):
            path = asset_path(*path.lstrip("/").split("/"))
        self._player.setSource(QUrl.fromLocalFile(path))

    # ===== 새 테마 스타일시트 (시안 반영) =====
    def _stylesheet(self) -> str:
        return """
        #glassPanel {
            background: rgba(255,255,255,1);
            border-radius: 28px;
            border: 1px solid rgba(255,255,255,0.25);
        }

        #TopBar {
            background: #1976d2;
            border-radius: 20px;
        }
        #Title {
            color: white;
            font-size: 44px;
            font-weight: 700;
            letter-spacing: 1px;
        }
        #BtnBack {
            background: rgba(255,255,255,0.18);
            color: white;
            border: none;
            padding: 0 22px;
            border-radius: 14px;
            font-size: 50px;
            font-weight: 500;
        }
        #BtnBack:hover { background: rgba(255,255,255,0.28); }
        #BtnUser {
            background: rgba(255,255,255,0.20);
            color: white;
            border: none;
            padding: 0 22px;
            border-radius: 14px;
            font-size: 28px;
            font-weight: 500;
        }
        #BtnUser:hover { background: rgba(255,255,255,0.28); }
        #BtnStart {
            background: #17c964;
            color: white;
            border: none;
            padding: 0 24px;
            border-radius: 14px;
            font-size: 28px;
            font-weight: 500;
        }
        #BtnStart:hover { background: #11b85a; }

        #LeftMenu {
            background: rgba(255,255,255,0.90);
            border-radius: 18px;
            min-width: 420px;
            max-width: 420px;
        }

        #VideoCard, #TipsCard {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(0,0,0,0.05);
            border-radius: 18px;
        }
        #VideoTitle, #TipsTitle {
            color: #111827;
            font-size: 36px;
            font-weight: 700;
        }
        """
