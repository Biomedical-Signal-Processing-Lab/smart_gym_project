# views/info_page.py
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
from db.models import User, WorkoutSession, SessionExercise
from sqlalchemy import func
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
    QGridLayout, QFrame, QProgressBar, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QPainter, QBrush
from core.page_base import PageBase
import pyqtgraph as pg

class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet("""
            QFrame#Card {
                background: #ffffff;
                border-radius: 16px;
                border: 1px solid #e8eef8;
            }
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

class Avatar(QWidget):
    def __init__(self, name="?", size=56, color="#7c8cf8", parent=None):
        super().__init__(parent)
        self.name = name; self.size_px = size; self.color = QColor(color)
        self.setFixedSize(QSize(size, size))
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self.color)); p.setPen(Qt.NoPen); p.drawEllipse(self.rect())
        init = (self.name[:1] if self.name else "?")
        p.setPen(QColor("white")); f = QFont("", int(self.size_px*0.38), QFont.Bold); p.setFont(f)
        p.drawText(self.rect(), Qt.AlignCenter, init)

class _NoMouseViewBox(pg.ViewBox):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.setMenuEnabled(False)
    def mouseClickEvent(self, ev):   ev.ignore()
    def mouseDragEvent(self, ev):    ev.ignore()
    def wheelEvent(self, ev):        ev.ignore()
    def contextMenuEvent(self, ev):  ev.ignore()

class InfoPage(PageBase):
    BASE_COLORS = [
        "#6aa7ff","#9b6bff","#19c37d","#f59e0b","#ef4444",
        "#22c55e","#06b6d4","#f472b6","#a3e635","#fb7185",
        "#f97316","#60a5fa","#a78bfa","#34d399","#4ade80"
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("InfoPage")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        pal = self.palette(); pal.setColor(QPalette.Window, QColor("#f5f8ff")); self.setPalette(pal)
        pg.setConfigOptions(antialias=True)

        self._ex_color = {}      
        self._stat_cards = {}    

        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget#InfoPage { background: #f5f8ff; }
            QLabel { color:#465065; font-size:16px; }
            QGroupBox { background:#fff; border:1px solid #e5ecf6; border-radius:16px; margin-top:16px; color:#3b3f53; font-weight:700; }
            QGroupBox::title { subcontrol-origin: margin; left:12px; padding:6px 10px; color:#556074; background:rgba(138,180,248,0.20); border-radius:10px; }
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #ffffff, stop:1 #eef4ff);
                          color:#334155; border:1px solid #d7e3f6; border-radius:12px; padding:10px 18px; font-weight:700; }
            QPushButton:hover { background:#f3f7ff; }
            .chip { background:#eef6ff; color:#24527a; border:1px solid #d6e8ff; padding:6px 10px; border-radius:10px; font-weight:600; }
            .muted { color:#6b7380; font-size:15px; }
        """)

        root = QVBoxLayout(self); root.setContentsMargins(18,18,18,18); root.setSpacing(14)

        header = QLabel("회원 정보")
        header.setStyleSheet("color:#0f172a; font-size:28px; font-weight:900; letter-spacing:0.4px;")
        header.setAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
        root.addWidget(header)

        grid = QGridLayout(); grid.setHorizontalSpacing(16); grid.setVerticalSpacing(16)

        grid.setColumnStretch(0, 1)  
        grid.setColumnStretch(1, 2)  
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 2)

        grid.setRowStretch(0, 1)     
        grid.setRowStretch(1, 3)     

        self.profile_card = Card()
        self.profile_card.setMaximumWidth(360)              
        p = QVBoxLayout(self.profile_card)
        p.setContentsMargins(18,16,18,16)                   
        p.setSpacing(8)                                     

        row = QHBoxLayout()
        self.avatar = Avatar("김", size=64)                 
        row.addWidget(self.avatar)

        namebox = QVBoxLayout()
        self.lbl_name = QLabel("—")
        self.lbl_name.setStyleSheet("font-size:26px; font-weight:900; color:#102a43;")  
        self.lbl_grade = QLabel("정회원")
        self.lbl_grade.setStyleSheet("color:#5b6a82; font-size:13px; font-weight:700;")
        namebox.addWidget(self.lbl_name)
        namebox.addWidget(self.lbl_grade)
        row.addLayout(namebox)

        p.addLayout(row)

        self.lbl_join  = QLabel("가입일 —");  self.lbl_join.setProperty("class","muted")
        self.lbl_until = QLabel("회원권 만료일 —"); self.lbl_until.setProperty("class","muted")

        self.lbl_last_workout = QLabel("마지막 운동일 —")
        self.lbl_last_workout.setProperty("class", "muted")
        self.lbl_last_workout.setStyleSheet("font-size:15px; color:#4b5563;")  

        p.addWidget(self.lbl_join)
        p.addWidget(self.lbl_until)
        p.addWidget(self.lbl_last_workout)             

        self.lbl_days_left = QLabel("남은 기간 —일")
        self.lbl_days_left.setStyleSheet("color:#425a70; font-size:16px; font-weight:700;")  
        self.pb_days = QProgressBar()
        self.pb_days.setTextVisible(False)
        self.pb_days.setFixedHeight(12)                   
        self.pb_days.setStyleSheet(
            "QProgressBar{background:#e9eef7; border-radius:6px;}"
            "QProgressBar::chunk{background:#7c8cf8; border-radius:6px;}"
        )
        p.addWidget(self.lbl_days_left)
        p.addWidget(self.pb_days)

        self.lbl_week_total = QLabel("이번 주 총 운동 횟수 0회")
        self.lbl_week_total.setStyleSheet("color:#1e40af; font-size:24px; font-weight:900;")  
        p.addWidget(self.lbl_week_total)


        grid.addWidget(self.profile_card, 0, 0, 2, 1)

        right_top = Card()
        rt = QVBoxLayout(right_top); rt.setContentsMargins(12,12,12,8); rt.setSpacing(8)

        nav = QHBoxLayout()
        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")
        for b in (self.btn_left, self.btn_right):
            b.setFixedWidth(48)
        nav.addStretch(1); nav.addWidget(self.btn_left); nav.addWidget(self.btn_right)
        rt.addLayout(nav)

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.scroll.setStyleSheet("QScrollArea{border:0; background:transparent;} QWidget{background:transparent;}")
        self.cards_container = QWidget(); self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(6,6,6,6); self.cards_layout.setSpacing(12)
        self.scroll.setWidget(self.cards_container)
        rt.addWidget(self.scroll)

        grid.addWidget(right_top, 0, 1, 1, 3)

        self.chart_card = Card()
        cc = QVBoxLayout(self.chart_card); cc.setContentsMargins(16,14,16,12); cc.setSpacing(6)
        title_row = QHBoxLayout()
        icon = QLabel("▴"); icon.setStyleSheet("color:#7c8cf8; font-size:16px;")
        t = QLabel("주간 운동 추이(꺾은선)"); t.setStyleSheet("color:#1f2937; font-size:16px; font-weight:800;")
        title_row.addWidget(icon); title_row.addWidget(t); title_row.addStretch(1)
        cc.addLayout(title_row)

        self.vb = _NoMouseViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb, background="#ffffff")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getPlotItem().setMenuEnabled(False)
        ax = self.plot.getAxis('bottom'); ax.setPen(pg.mkPen('#9fb0c6'))
        ax = self.plot.getAxis('left');   ax.setPen(pg.mkPen('#9fb0c6'))
        self.plot.setLabel('left', "횟수", **{'color':'#5b6a82', 'size':'12pt'})
        self.plot.setLabel('bottom', "요일", **{'color':'#5b6a82', 'size':'12pt'})
        cc.addWidget(self.plot, 1)

        grid.addWidget(self.chart_card, 1, 1, 1, 3)
        root.addLayout(grid)

        btn_row = QHBoxLayout()
        self.btn_back = QPushButton("운동 화면으로")
        self.btn_logout = QPushButton("로그아웃")
        btn_row.addStretch(1); btn_row.addWidget(self.btn_back); btn_row.addWidget(self.btn_logout)
        root.addLayout(btn_row)

        self.btn_back.clicked.connect(lambda: self._goto("select"))
        self.btn_logout.clicked.connect(self._logout)
        self.btn_left.clicked.connect(lambda: self._scroll_stats(-1))
        self.btn_right.clicked.connect(lambda: self._scroll_stats(+1))

    def on_enter(self, ctx):
        self.ctx = ctx
        self._refresh()

    def _refresh(self):
        if not self.ctx.is_logged_in():
            self._show_logged_out_view(); return
        try:
            with self.ctx.SessionLocal() as s:
                user = s.query(User).filter_by(id=self.ctx.current_user_id).one_or_none()
                if not user:
                    self.ctx.clear_current_user(); self._show_logged_out_view(); return

                self.lbl_name.setText(f"{user.name}")
                self.avatar.name = (user.name or "?"); self.avatar.update()
                self.lbl_join.setText(f"가입일  {self._fmt_date(getattr(user,'created_at',None))}")

                today = datetime.now().date()
                mock_until = today + timedelta(days=54)  

                self.lbl_until.setText(f"회원권 만료일  {mock_until.strftime('%Y년 %m월 %d일')}")
                days_left = (mock_until - today).days
                self.lbl_days_left.setText(f"남은 기간  {days_left}일")

                self.pb_days.setMaximum(days_left if days_left > 0 else 1)
                self.pb_days.setValue(max(0, days_left))

                today = datetime.now().date()
                start_date = today - timedelta(days=6)
                date_expr = func.date(WorkoutSession.started_at)  
                rows = (
                    s.query(
                        date_expr.label("d"),
                        SessionExercise.exercise_name.label("ex"),
                        func.sum(SessionExercise.reps).label("cnt"),
                        func.avg(SessionExercise.avg_score).label("avg"),
                    )
                    .join(SessionExercise, SessionExercise.session_id == WorkoutSession.id)
                    .filter(WorkoutSession.user_id == user.id)
                    .filter(WorkoutSession.started_at.isnot(None)) 
                    .filter(date_expr >= str(start_date))
                    .group_by("d", "ex")
                    .all()
                )

                days = [(start_date + timedelta(days=i)).isoformat() for i in range(7)]
                by_day = defaultdict(lambda: defaultdict(int))
                for d, ex, cnt, _ in rows:
                    by_day[str(d)][ex] = int(cnt or 0)

                week_total = sum(sum(by_day[d].values()) for d in days)
                self.lbl_week_total.setText(f"이번 주 총 운동 횟수  {week_total:,}회")

                stats = (
                    s.query(
                        SessionExercise.exercise_name.label("ex"),
                        func.sum(SessionExercise.reps).label("total"),
                        func.avg(SessionExercise.avg_score).label("avg"),
                    )
                    .join(WorkoutSession, SessionExercise.session_id == WorkoutSession.id)
                    .filter(WorkoutSession.user_id == user.id)
                    .group_by("ex")
                    .all()
                )
                stat_map = {ex: (int(total or 0), float(avg or 0)) for ex, total, avg in stats}

                self._ensure_colors(list(stat_map.keys()))
                self._rebuild_stat_cards(stat_map)
                self._render_line_chart(days, by_day)

        except Exception:
            traceback.print_exc()
            QMessageBox.critical(self, "오류", "정보를 불러오지 못했습니다.")
            self._show_profile_only()

    def _rebuild_stat_cards(self, stat_map):
        for i in reversed(range(self.cards_layout.count())):
            w = self.cards_layout.itemAt(i).widget()
            if w: w.deleteLater()
        self._stat_cards.clear()

        for ex, (total, avg) in sorted(stat_map.items(), key=lambda x: -x[1][0]):
            color = self._ex_color.get(ex, "#6aa7ff")
            card = self._make_mini_card(ex, color, total, avg)
            self.cards_layout.addWidget(card)
            self._stat_cards[ex] = card

        self.cards_layout.addStretch(1)

    def _make_mini_card(self, ex, color, total, avg):
        w = Card()
        lay = QVBoxLayout(w); lay.setContentsMargins(16,14,16,14); lay.setSpacing(6)
        row = QHBoxLayout()
        dot = QLabel("●"); dot.setStyleSheet(f"color:{color}; font-size:14px;")
        ttl = QLabel(self._ex_display(ex)); ttl.setStyleSheet("color:#5b6a82; font-size:14px; font-weight:700;")
        row.addWidget(dot); row.addWidget(ttl); row.addStretch(1)
        lay.addLayout(row)
        total_lbl = QLabel(f"{total:,}회"); total_lbl.setStyleSheet("color:#111827; font-size:26px; font-weight:800;")
        avg_lbl = QLabel(f"평균 점수  {avg:.0f} 점"); avg_lbl.setStyleSheet("color:#8894a8; font-size:13px;")
        lay.addWidget(total_lbl); lay.addWidget(avg_lbl); lay.addStretch(1)
        w.setFixedWidth(220)
        return w

    def _scroll_stats(self, direction: int):
        bar = self.scroll.horizontalScrollBar()
        step = int(self.scroll.viewport().width() * 0.8)
        bar.setValue(max(0, bar.value() + (step * direction)))

    def _render_line_chart(self, days, by_day):
        self.plot.clear()

        self.legend = self.plot.addLegend(offset=(10, 10))
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 220))
        self.legend.setPen(pg.mkPen('#e5ecf6'))

        xs = list(range(len(days)))
        xlabels = [d[5:] for d in days]  
        self.plot.getAxis('bottom').setTicks([list(zip(xs, xlabels))])

        exercises = sorted({ex for d in days for ex in by_day[d].keys()})
        if not exercises:
            return

        ymax = 0
        for ex in exercises:
            color = self._ex_color.get(ex) or self._assign_color(ex)
            pen = pg.mkPen(color, width=3)
            base = pg.mkColor(color)
            brush = pg.mkBrush(base.red(), base.green(), base.blue(), 55)

            ys = [by_day[d].get(ex, 0) for d in days]
            ymax = max(ymax, max(ys or [0]))

            self.plot.plot(
                xs, ys,
                pen=pen,
                name=self._ex_display(ex),
                antialias=True,
                fillLevel=0, 
                brush=brush
            )

            scatter = pg.ScatterPlotItem(
                size=9, brush=pg.mkBrush(color), pen=pg.mkPen('#ffffff', width=2)
            )
            scatter.addPoints([{'pos': (i, y)} for i, y in enumerate(ys)])
            self.plot.addItem(scatter)

        vb = self.plot.getViewBox()
        vb.setXRange(-0.25, len(days) - 0.75, padding=0.02)
        vb.setYRange(0, max(5, ymax) * 1.15 if ymax > 0 else 5, padding=0.12)

    def _fill_path(self, xs, ys):
        from PySide6.QtGui import QPainterPath
        from PySide6.QtCore import QPointF
        if not xs: return None
        p = QPainterPath(); p.moveTo(QPointF(xs[0], 0))
        for x, y in zip(xs, ys): p.lineTo(QPointF(x, y))
        p.lineTo(QPointF(xs[-1], 0)); p.closeSubpath()
        return p

    def _ensure_colors(self, exercises):
        i = 0
        for ex in exercises:
            if ex not in self._ex_color:
                self._ex_color[ex] = self.BASE_COLORS[i % len(self.BASE_COLORS)]
                i += 1

    def _assign_color(self, ex):
        c = self.BASE_COLORS[len(self._ex_color) % len(self.BASE_COLORS)]
        self._ex_color[ex] = c; return c

    def _ex_display(self, key):  
        mapping = {"squat":"스쿼트","lunge":"런지","plank":"플랭크"}
        return mapping.get(key, key)

    def _fmt_date(self, dt):
        if not dt: return "—"
        try:
            if isinstance(dt, str): return dt.split("T")[0]
            return dt.strftime("%Y년 %m월 %d일")
        except Exception:
            return str(dt)

    def _show_profile_only(self):
        try: self.plot.clear()
        except Exception: pass

    def _show_logged_out_view(self):
        self.lbl_name.setText("—")
        self.lbl_join.setText("가입일 —")
        self.lbl_until.setText("회원권 만료일 —")
        self.lbl_days_left.setText("남은 기간 —일")
        self.pb_days.setValue(0)
        self.btn_logout.setEnabled(False)
        ret = QMessageBox.question(self, "안내", "회원가입 하시겠습니까?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ret == QMessageBox.Yes: self._goto("enroll")
        else: self._goto("start")

    def _logout(self):
        self.ctx.clear_current_user()
        QMessageBox.information(self, "로그아웃", "로그아웃되었습니다.")
        self._goto("start")

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"): router = router.parent()
        if router: router.navigate(page)
