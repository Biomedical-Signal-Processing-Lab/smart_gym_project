# views/info_page.py
import traceback
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox
from PySide6.QtCore import Qt
from core.page_base import PageBase
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sqlalchemy import func

class InfoPage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("InfoPage")

        self.lbl_name = QLabel("-")
        self.lbl_created = QLabel("-")

        prof_group = QGroupBox("프로필")
        prof_v = QVBoxLayout()
        prof_v.addWidget(self.lbl_name)
        prof_v.addWidget(self.lbl_created)
        prof_group.setLayout(prof_v)

        self.chart_group = QGroupBox("최근 7일 운동 활동")
        cg_layout = QVBoxLayout(self.chart_group)

        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        cg_layout.addWidget(self.canvas)

        self.chart_empty = QLabel("최근 7일 데이터가 없습니다.")
        self.chart_empty.setAlignment(Qt.AlignCenter)
        self.chart_empty.setStyleSheet("color: #aaa;")
        cg_layout.addWidget(self.chart_empty)
        self.chart_empty.hide()

        self.btn_back = QPushButton("뒤로가기")
        self.btn_logout = QPushButton("로그아웃")
        self.btn_delete = QPushButton("회원탈퇴")

        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(self.btn_back)
        btns.addWidget(self.btn_logout)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)

        root = QVBoxLayout(self)
        root.addWidget(prof_group, 1)     
        root.addWidget(self.chart_group, 1)  
        root.addLayout(btns)

        self.btn_back.clicked.connect(lambda: self._goto("select"))
        self.btn_logout.clicked.connect(self._logout)
        self.btn_delete.clicked.connect(self._delete_account)

    def on_enter(self, ctx):
        self.ctx = ctx
        self._refresh()

    def _refresh(self):
        if not self.ctx.is_logged_in():
            self._show_logged_out_view()
            return

        try:
            with self.ctx.SessionLocal() as s:
                from db.models import User
                user = s.query(User).filter_by(id=self.ctx.current_user_id).one_or_none()
                if not user:
                    self.ctx.clear_current_user()
                    self._show_logged_out_view()
                    return

                self.lbl_name.setText(f"이름: {user.name}")
                self.lbl_created.setText(f"가입일: {user.created_at}")

                self.btn_logout.setEnabled(True)
                self.btn_delete.setEnabled(True)

                self._render_activity_chart(s, user.id)

        except Exception:
            traceback.print_exc()
            QMessageBox.critical(self, "오류", "정보를 불러오지 못했습니다.")
            self._show_profile_only()

    def _render_activity_chart(self, s, user_id: int):
        from db.models import WorkoutSession, SessionExercise
        from sqlalchemy import func
        import numpy as np
        from datetime import datetime, timedelta

        today = datetime.now().date()
        start_date = today - timedelta(days=6)

        date_expr = func.date(WorkoutSession.started_at)

        rows = (
            s.query(
                date_expr.label("d"),
                SessionExercise.exercise_name.label("ex"),
                func.sum(SessionExercise.reps).label("cnt"),
            )
            .join(SessionExercise, SessionExercise.session_id == WorkoutSession.id)
            .filter(WorkoutSession.user_id == user_id)
            .filter(date_expr >= str(start_date))
            .group_by("d", "ex")
            .all()
        )

        data = {}
        exercises = set()
        for d, ex, cnt in rows:
            exercises.add(ex)
            data.setdefault(d, {})[ex] = int(cnt or 0)

        days = [(start_date + timedelta(days=i)).isoformat() for i in range(7)]

        preferred = ["squat", "lunge", "pushup", "plank"]
        ex_list = [ex for ex in preferred if ex in exercises] + \
                [ex for ex in sorted(exercises) if ex not in preferred]

        values = {ex: [data.get(day, {}).get(ex, 0) for day in days] for ex in ex_list}
        totals = [sum(values[ex][i] for ex in ex_list) for i in range(len(days))]

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_title("exercise history")
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels([d[5:] for d in days])
        ax.set_ylabel("count")

        if not ex_list or all(t == 0 for t in totals):
            self.chart_empty.show()
            ax.plot(range(len(days)), [0]*len(days), linestyle="--")
            self.fig.tight_layout()
            self.canvas.draw_idle()
            return
        else:
            self.chart_empty.hide()

        x = np.arange(len(days))
        for ex in ex_list:
            y = np.array(values[ex], dtype=float)
            ax.plot(x, y, marker="o", label=ex)

        ax.legend(loc="upper left")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _show_profile_only(self):
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.set_title("exercise history")
            ax.set_ylabel("count")
            ax.plot([], [])
            self.chart_empty.setText("데이터를 가져올 수 없습니다.")
            self.chart_empty.show()
            self.fig.tight_layout()
            self.canvas.draw_idle()
        except Exception:
            pass

    def _show_logged_out_view(self):
        self.lbl_name.setText("이름: -")
        self.lbl_created.setText("가입일: -")

        self.btn_logout.setEnabled(False)
        self.btn_delete.setEnabled(False)

        ret = QMessageBox.question(
            self, "안내", "로그인 하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if ret == QMessageBox.Yes:
            self._goto("login")
        else:
            self._goto("start")

    def _logout(self):
        self.ctx.clear_current_user()
        QMessageBox.information(self, "로그아웃", "로그아웃되었습니다.")
        self._goto("start")

    def _delete_account(self):
        if not self.ctx.is_logged_in():
            return
        ret = QMessageBox.warning(
            self, "회원탈퇴",
            "정말로 계정을 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return

        try:
            with self.ctx.SessionLocal() as s:
                from db.models import User
                user = s.query(User).filter_by(id=self.ctx.current_user_id).one_or_none()
                if user:
                    s.delete(user)
                    s.commit()

            self.ctx.clear_current_user()
            QMessageBox.information(self, "완료", "계정이 삭제되었습니다.")
            self._goto("start")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"삭제 실패: {e}")

    def _goto(self, page: str):
        router = self.parent()
        while router and not hasattr(router, "navigate"):
            router = router.parent()
        if router:
            router.navigate(page)
