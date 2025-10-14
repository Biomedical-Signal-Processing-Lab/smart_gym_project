from typing import List, Optional, Iterable
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFrame, QGridLayout, QSizePolicy
)
from core.page_base import PageBase
from data.guide_data import Exercise, list_all
from ui.guide_style import (
    style_page_root, style_side_panel, style_scrollarea, enable_touch_scroll, style_exercise_card, 
    style_info_card, style_header_title, style_header_chip, style_header_desc, force_bg
)

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
        root.addWidget(sets)

        force_bg(self, """
            QFrame#ExerciseCard { background:#ffffff; border:1px solid #e5ecf6; border-radius:14px; }
            QFrame#ExerciseCard[selected="true"] { border:2px solid #7aa2ff; }
            QLabel#title { font-size:20px; font-weight:700; color:#1f2937; }
            QLabel#sets  { color:#6b7380; }
            QLabel#chip  { background:#eef6ff; border:1px solid #d6e8ff; border-radius:999px;
                           padding:2px 8px; color:#24527a; font-size:16px; font-weight:600; }
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
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)

        ttl = QLabel(title)
        ttl.setObjectName("CardTitle")
        ttl.setStyleSheet("background:transparent; color:#1f2937; font-weight:800;")
        lay.addWidget(ttl)

        def paint_texts(w: QWidget):
            if isinstance(w, QLabel):
                w.setStyleSheet("background:transparent; color:#374151;")
            for ch in w.findChildren(QLabel):
                ch.setStyleSheet("background:transparent; color:#374151;")

        paint_texts(body_widget)
        lay.addWidget(body_widget)
        style_info_card(self)

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
        dot.setStyleSheet("background:transparent; color:#4b5563;")
        dot.setFixedWidth(18)
        lbl = QLabel(t)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("background:transparent; color:#374151;")
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
        style_page_root(self)

        self.exercises = list_all()

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        hdr = QLabel("운동 가이드")
        hdr.setObjectName("PageTitle")
        hdr.setStyleSheet("font-size:80px; font-weight:1000; color:#0f172a; background:transparent;")
        root.addWidget(hdr)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(16)
        root.addLayout(body, 1)

        self.left_panel = self._build_left_panel()
        body.addWidget(self.left_panel, 0)

        self.detail_panel = self._build_detail_panel()
        body.addWidget(self.detail_panel, 1)

        if self.exercises:
            self._select(self.exercises[0])

    def _build_left_panel(self) -> QWidget:
        side = QFrame()
        style_side_panel(side)
        side.setFixedWidth(340)

        v = QVBoxLayout(side)
        v.setContentsMargins(0, 0, 16, 0)
        v.setSpacing(8)

        scroll = QScrollArea()
        style_scrollarea(scroll)
        enable_touch_scroll(scroll, mouse_drag=True)

        content = QWidget()
        force_bg(content, "background:#f3f9ff;")
        lv = QVBoxLayout(content)
        lv.setContentsMargins(8, 8, 4, 8)   
        lv.setSpacing(8)                    

        self._cards: List[ExerciseCard] = []
        for ex in self.exercises:
            card = ExerciseCard(ex)
            style_exercise_card(card)
            card.mousePressEvent = lambda e, _ex=ex: self._select(_ex)
            self._cards.append(card)
            lv.addWidget(card)

        lv.addStretch(1)

        scroll.setWidget(content)
        v.addWidget(scroll, 1)
        return side

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget()
        force_bg(panel, "background:#f3f9ff;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QWidget()
        force_bg(header, "background:transparent;")
        hb = QVBoxLayout(header)
        hb.setContentsMargins(4, 0, 4, 8)
        hb.setSpacing(6)

        first_line = QWidget(header)
        force_bg(first_line, "background:transparent;")
        fl = QHBoxLayout(first_line)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(10)

        self.h_title = QLabel("", first_line)
        style_header_title(self.h_title)

        self.h_cate = QLabel("")
        style_header_chip(self.h_cate)

        fl.addWidget(self.h_title, 0, Qt.AlignVCenter)
        fl.addWidget(self.h_cate, 0, Qt.AlignVCenter)
        fl.addStretch(1)

        self.h_desc = QLabel("")
        self.h_desc.setTextFormat(Qt.PlainText)
        self.h_desc.setWordWrap(True)
        style_header_desc(self.h_desc)

        hb.addWidget(first_line)
        hb.addWidget(self.h_desc)
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.v_goal = QLabel("")
        self.v_goal.setWordWrap(True)
        self.v_goal.setStyleSheet("background:transparent;")

        self.v_reco = QLabel("")
        self.v_reco.setWordWrap(True)
        self.v_reco.setStyleSheet("background:transparent; font-size:24px; font-weight:800; color:#0f172a;")

        self.v_cate = QLabel("")
        self.v_cate.setWordWrap(True)
        self.v_cate.setStyleSheet("background:transparent;")

        grid.addWidget(InfoCard("목표 근육", self.v_goal), 0, 0)
        grid.addWidget(InfoCard("권장 운동량", self.v_reco), 0, 1)
        grid.addWidget(InfoCard("운동 분류", self.v_cate), 0, 2)
        layout.addLayout(grid)

        self.steps_widget = bullet_list([], numbered=True)
        layout.addWidget(InfoCard("운동 방법", self.steps_widget))

        self.tips_widget = bullet_list([], numbered=False)
        layout.addWidget(InfoCard("주의사항 및 팁", self.tips_widget))

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(0)

        btn_box = QWidget()
        bb = QHBoxLayout(btn_box)
        bb.setContentsMargins(0, 0, 0, 0)
        bb.setSpacing(0)

        self.btn_profile = IconButton("app/assets/btn_info.png", size=self.ICON_SIZE)
        self.btn_profile.clicked.connect(lambda: self._goto("info"))

        self.btn_start = IconButton("app/assets/btn_start.png", size=self.ICON_SIZE)
        self.btn_start.clicked.connect(lambda: self._goto("exercise"))

        bb.addWidget(self.btn_profile)
        bb.addWidget(self.btn_start)

        btn_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        footer.addStretch(1)
        footer.addWidget(btn_box, 0, Qt.AlignRight | Qt.AlignBottom)

        layout.addStretch(1)
        layout.addLayout(footer)
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
            dot.setFixedWidth(18)
            dot.setStyleSheet("background:transparent; color:#4b5563;")
            lbl = QLabel(t)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("background:transparent; color:#374151;")
            row.addWidget(dot)
            row.addWidget(lbl, 1)
            lay.addLayout(row)

    def _goto(self, page: str) -> None:
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)
