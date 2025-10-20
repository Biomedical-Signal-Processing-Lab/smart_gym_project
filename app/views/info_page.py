# views/info_page.py
import traceback
from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import func
from db.models import User, WorkoutSession, SessionExercise

from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
    QGridLayout, QFrame, QProgressBar, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QPainter, QBrush

from core.page_base import PageBase
import pyqtgraph as pg

from ui.info_style import apply_info_page_styles

class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

class Avatar(QWidget):
    """원형 배경 위에 이니셜 한 글자."""
    def __init__(self, name="?", size=56, color="#7c8cf8", parent=None):
        super().__init__(parent)
        self.name = name
        self.size_px = size
        self.color = QColor(color)
        self.setFixedSize(QSize(size, size))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self.color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(self.rect())
        init = (self.name[:1] if self.name else "?")
        p.setPen(QColor("white"))
        f = QFont("", int(self.size_px * 0.42), QFont.Bold)
        p.setFont(f)
        p.drawText(self.rect(), Qt.AlignCenter, init)

class _NoMouseViewBox(pg.ViewBox):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setMenuEnabled(False)

    def mouseClickEvent(self, ev):   ev.ignore()
    def mouseDragEvent(self, ev):    ev.ignore()
    def wheelEvent(self, ev):        ev.ignore()
    def contextMenuEvent(self, ev):  ev.ignore()

class InfoPage(PageBase):
    BASE_COLORS = [
        "#6aa7ff", "#9b6bff", "#19c37d", "#f59e0b", "#ef4444",
        "#22c55e", "#06b6d4", "#f472b6", "#a3e635", "#fb7185",
        "#f97316", "#60a5fa", "#a78bfa", "#34d399", "#4ade80"
    ]

    def __init__(self):
        super().__init__()
        self.setObjectName("InfoPage")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor("#f5f8ff"))
        self.setPalette(pal)

        pg.setConfigOptions(antialias=True)

        self._ex_color = {}
        self._stat_cards = {}

        self._build_ui()

    def _build_ui(self):
        # 공통 스타일 적용
        apply_info_page_styles(self)

        # 페이지 배경 이미지 적용 (프로젝트 경로 기준)
        self.setStyleSheet(self.styleSheet() + """
            QWidget#InfoPage {
                background: #f5f8ff;
                background-image: url(assets/background/bg_gym.jpg);
                background-position: center;
                background-repeat: no-repeat;
                background-origin: content;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        # ── 상단 보라색 패널(좌: 타이틀, 우: 버튼) ─────────────────────────
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_bar.setStyleSheet("""
            QFrame#TopBar {
                background: rgba(126, 58, 242, 0.92); /* 보라색 */
                border-radius:12px;
            }
            QLabel#TopTitle {
                color: #ffffff;
                font-size: 40px; /* 큰 타이틀 */
                font-weight: 700;
                padding-left: 4px;
            }
            QPushButton#BtnPrimary {
                background:#2563eb; color:white;
                border:1px solid #1d4ed8;
                border-radius:10px; padding:10px 18px;
                font-size:25px; font-weight:500;
            }
            QPushButton#BtnPrimary:hover { background:#1d4ed8; }
            QPushButton#BtnDanger {
                background:#ef4444; color:white;
                border:1px solid #dc2626;
                border-radius:10px; padding:10px 18px;
                font-size:25px; font-weight:500;
            }
            QPushButton#BtnDanger:hover { background:#dc2626; }
        """)
        tb = QHBoxLayout(top_bar)
        tb.setContentsMargins(16, 10, 16, 10)
        # 좌측 타이틀
        self.top_title = QLabel("회원 정보")
        self.top_title.setObjectName("TopTitle")
        tb.addWidget(self.top_title, 0, Qt.AlignVCenter)
        tb.addStretch(1)
        # 우측 버튼들
        self.btn_back = QPushButton("운동 화면으로")
        self.btn_back.setObjectName("BtnPrimary")
        self.btn_logout = QPushButton("로그아웃")
        self.btn_logout.setObjectName("BtnDanger")
        tb.addWidget(self.btn_back)
        tb.addWidget(self.btn_logout)
        root.addWidget(top_bar)

        # 그리드
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 2)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 3)

        # 좌측 프로필 카드
        self.profile_card = Card()
        self.profile_card.setMaximumWidth(360)
        p = QVBoxLayout(self.profile_card)
        p.setContentsMargins(18, 16, 18, 16)
        p.setSpacing(10)

        # 상단: 아바타 + 이름(크게)
        row = QHBoxLayout()
        self.avatar = Avatar("김", size=64)
        row.addWidget(self.avatar)

        namebox = QVBoxLayout()
        self.lbl_name = QLabel("—")
        self.lbl_name.setProperty("cls", "display")
        self.lbl_grade = QLabel("정회원")
        self.lbl_grade.setProperty("cls", "muted")
        namebox.addWidget(self.lbl_name)
        namebox.addWidget(self.lbl_grade)
        row.addLayout(namebox)
        p.addLayout(row)

        # 가입/만료/마지막 운동일(제목-날짜)
        self.lbl_join_cap = QLabel("가입일")
        self.lbl_join_cap.setProperty("cls", "muted")
        self.lbl_join_date = QLabel("—")
        self.lbl_join_date.setProperty("cls", "date")

        self.lbl_until_cap = QLabel("회원권 만료일")
        self.lbl_until_cap.setProperty("cls", "muted")
        self.lbl_until_date = QLabel("—")
        self.lbl_until_date.setProperty("cls", "date")

        self.lbl_last_workout_cap = QLabel("마지막 운동일")
        self.lbl_last_workout_cap.setProperty("cls", "muted")
        self.lbl_last_workout = QLabel("—")
        self.lbl_last_workout.setProperty("cls", "date")

        p.addWidget(self.lbl_join_cap)
        p.addWidget(self.lbl_join_date)
        p.addSpacing(4)
        p.addWidget(self.lbl_until_cap)
        p.addWidget(self.lbl_until_date)
        p.addSpacing(4)
        p.addWidget(self.lbl_last_workout_cap)
        p.addWidget(self.lbl_last_workout)

        # 남은 기간 + 진행바
        self.lbl_days_left = QLabel("남은 기간 —일")
        p.addWidget(self.lbl_days_left)

        self.pb_days = QProgressBar()
        self.pb_days.setTextVisible(False)
        self.pb_days.setFixedHeight(12)
        p.addWidget(self.pb_days)

        # 주간 합계
        self.lbl_week_total = QLabel("이번 주 총 운동 횟수 0회")
        self.lbl_week_total.setProperty("cls", "title")
        p.addWidget(self.lbl_week_total)

        grid.addWidget(self.profile_card, 0, 0, 2, 1)

        # 우상단 통계 카드 리스트
        right_top = Card()
        rt = QVBoxLayout(right_top)
        rt.setContentsMargins(12, 12, 12, 8)
        rt.setSpacing(8)

        nav = QHBoxLayout()
        self.btn_left = QPushButton("◀")
        self.btn_right = QPushButton("▶")
        for b in (self.btn_left, self.btn_right):
            b.setFixedWidth(20)
        nav.addStretch(1)
        nav.addWidget(self.btn_left)
        nav.addWidget(self.btn_right)
        rt.addLayout(nav)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea{border:0; background:transparent;} QWidget{background:transparent;}")

        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(6, 6, 6, 6)
        self.cards_layout.setSpacing(12)
        self.scroll.setWidget(self.cards_container)
        rt.addWidget(self.scroll)

        grid.addWidget(right_top, 0, 1, 1, 3)

        # 하단 차트 카드
        self.chart_card = Card()
        cc = QVBoxLayout(self.chart_card)
        cc.setContentsMargins(16, 14, 16, 12)
        cc.setSpacing(6)

        title_row = QHBoxLayout()
        icon = QLabel("▴")
        icon.setProperty("cls", "icon")
        t = QLabel("주간 운동 추이")
        t.setProperty("cls", "title")
        title_row.addWidget(icon)
        title_row.addWidget(t)
        title_row.addStretch(1)
        cc.addLayout(title_row)

        self.vb = _NoMouseViewBox()
        self.plot = pg.PlotWidget(viewBox=self.vb, background="#ffffff")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getPlotItem().setMenuEnabled(False)
        ax = self.plot.getAxis('bottom'); ax.setPen(pg.mkPen('#9fb0c6'))
        ax = self.plot.getAxis('left');   ax.setPen(pg.mkPen('#9fb0c6'))

        # 차트 라벨: 20pt(≈26~27px)
        self.plot.setLabel('left', "횟수", **{'color': '#5b6a82', 'size': '20pt'})
        self.plot.setLabel('bottom', "요일", **{'color': '#5b6a82', 'size': '20pt'})

        # 눈금 폰트
        tick_font = QFont()
        tick_font.setPointSize(18)        # ≈ 24px
        tick_font.setWeight(QFont.Medium) # 500
        self.plot.getAxis('bottom').setStyle(tickFont=tick_font)
        self.plot.getAxis('left').setStyle(tickFont=tick_font)

        cc.addWidget(self.plot, 1)

        grid.addWidget(self.chart_card, 1, 1, 1, 3)
        root.addLayout(grid)

        # 시그널
        self.btn_back.clicked.connect(lambda: self._goto("guide"))
        self.btn_logout.clicked.connect(self._logout)
        self.btn_left.clicked.connect(lambda: self._scroll_stats(-1))
        self.btn_right.clicked.connect(lambda: self._scroll_stats(+1))

    def on_enter(self, ctx):
        self.ctx = ctx
        self._refresh()

    def _refresh(self):
        if not self.ctx.is_logged_in():
            self._show_logged_out_view()
            return
        try:
            with self.ctx.SessionLocal() as s:
                user = s.query(User).filter_by(id=self.ctx.current_user_id).one_or_none()
                if not user:
                    self.ctx.clear_current_user()
                    self._show_logged_out_view()
                    return

                # 프로필 텍스트
                self.lbl_name.setText(f"{user.name}")
                if hasattr(self, "avatar"):
                    self.avatar.name = (user.name or "?")
                    self.avatar.update()

                # 가입일 / 만료일 / 마지막 운동일
                self.lbl_join_date.setText(self._fmt_date(getattr(user, 'created_at', None)))

                today = datetime.now().date()
                mock_until = today + timedelta(days=54)
                self.lbl_until_date.setText(mock_until.strftime('%Y년 %m월 %d일'))

                # 마지막 운동일(최근 세션)
                last_row = (
                    s.query(func.max(WorkoutSession.started_at))
                    .filter(WorkoutSession.user_id == user.id)
                    .one()
                )
                last_dt = last_row[0] if last_row and last_row[0] else None
                self.lbl_last_workout.setText(self._fmt_date(last_dt))

                # 남은 일수/진행바
                days_left = (mock_until - today).days
                self.lbl_days_left.setText(f"남은 기간  {days_left}일")
                self.pb_days.setMaximum(days_left if days_left > 0 else 1)
                self.pb_days.setValue(max(0, days_left))

                # ── 7일치 집계(실제 데이터만 표시) ───────────────────────────
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

                # 전체 운동별 통계 (실제 존재하는 운동만 카드로)
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
        # 초기화
        for i in reversed(range(self.cards_layout.count())):
            w = self.cards_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        self._stat_cards.clear()

        # 카드 생성(총 횟수 내림차순)
        for ex, (total, avg) in sorted(stat_map.items(), key=lambda x: -x[1][0]):
            color = self._ex_color.get(ex, "#6aa7ff")
            card = self._make_mini_card(ex, color, total, avg)
            self.cards_layout.addWidget(card)
            self._stat_cards[ex] = card

        self.cards_layout.addStretch(1)

    def _make_mini_card(self, ex, color, total, avg):
        w = Card()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)

        row = QHBoxLayout()
        dot = QLabel("●")
        dot.setProperty("cls", "icon")
        dot.setStyleSheet(f"color:{color};")
        ttl = QLabel(self._ex_display(ex))
        ttl.setProperty("cls", "muted")

        row.addWidget(dot)
        row.addWidget(ttl)
        row.addStretch(1)
        lay.addLayout(row)

        total_lbl = QLabel(f"{total:,}회")
        total_lbl.setProperty("cls", "kpi")

        avg_lbl = QLabel(f"평균 점수  {avg:.0f} 점")
        avg_lbl.setProperty("cls", "muted")

        lay.addWidget(total_lbl)
        lay.addWidget(avg_lbl)
        lay.addStretch(1)

        w.setFixedWidth(220)
        return w

    def _scroll_stats(self, direction: int):
        bar = self.scroll.horizontalScrollBar()
        step = int(self.scroll.viewport().width() * 0.8)
        bar.setValue(max(0, bar.value() + (step * direction)))

    def _render_line_chart(self, days, by_day):
        self.plot.clear()

        self.legend = self.plot.addLegend(offset=(10, 10), labelTextSize="19pt")
        self.legend.setBrush(pg.mkBrush(255, 255, 255, 220))
        self.legend.setPen(pg.mkPen('#e5ecf6'))

        xs = list(range(len(days)))
        xlabels = [d[5:] for d in days]  # MM-DD
        self.plot.getAxis('bottom').setTicks([list(zip(xs, xlabels))])

        # 실제 데이터에 존재하는 운동만 시리즈로
        exercises = sorted({ex for d in days for ex in by_day[d].keys()})
        if not exercises:
            vb = self.plot.getViewBox()
            vb.setXRange(-0.25, max(0, len(days) - 0.75), padding=0.02)
            vb.setYRange(0, 5, padding=0.12)
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
        if not xs:
            return None
        p = QPainterPath()
        p.moveTo(QPointF(xs[0], 0))
        for x, y in zip(xs, ys):
            p.lineTo(QPointF(x, y))
        p.lineTo(QPointF(xs[-1], 0))
        p.closeSubpath()
        return p

    def _ensure_colors(self, exercises):
        i = 0
        for ex in exercises:
            if ex not in self._ex_color:
                self._ex_color[ex] = self.BASE_COLORS[i % len(self.BASE_COLORS)]
                i += 1

    def _assign_color(self, ex):
        c = self.BASE_COLORS[len(self._ex_color) % len(self.BASE_COLORS)]
        self._ex_color[ex] = c
        return c

    def _ex_display(self, key):
        mapping = {
            "squat": "스쿼트",
            "leg_raise": "레그 레이즈",
            "pushup": "푸시업",
            "shoulder_press": "숄더 프레스",
            "side_lateral_raise": "사레레",
            "Bentover_Dumbbell": "덤벨 로우",
            "bentover_dumbbell": "덤벨 로우",
            "burpee": "버피",
            "jumping_jack": "점핑잭",
        }
        return mapping.get(key, key)

    def _fmt_date(self, dt):
        if not dt:
            return "—"
        try:
            if isinstance(dt, str):
                return dt.split("T")[0]
            return dt.strftime("%Y년 %m월 %d일")
        except Exception:
            return str(dt)

    def _show_profile_only(self):
        try:
            self.plot.clear()
        except Exception:
            pass

    def _show_logged_out_view(self):
        self.lbl_name.setText("—")
        self.lbl_join_date.setText("—")
        self.lbl_until_date.setText("—")
        self.lbl_last_workout.setText("—")
        self.pb_days.setValue(0)
        self.btn_logout.setEnabled(False)
        ret = QMessageBox.question(
            self, "안내", "회원가입 하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            self._goto("enroll")
        else:
            self._goto("start")

    def _logout(self):
        self.ctx.clear_current_user()
        QMessageBox.information(self, "로그아웃", "로그아웃되었습니다.")
        self._goto("start")

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)
