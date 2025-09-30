# app/views/info_page.py
import traceback
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QGroupBox
)
from PySide6.QtCore import Qt
from core.page_base import PageBase

class InfoPage(PageBase):
    def __init__(self):
        super().__init__()
        self.setObjectName("InfoPage")

        self.title = QLabel("내 정보")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 24px; font-weight: 800;")

        self.lbl_name = QLabel("-")
        self.lbl_created = QLabel("-")
        self.lbl_emb_count = QLabel("-")

        for l in (self.lbl_name, self.lbl_created, self.lbl_emb_count):
            l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet("font-size: 16px;")

        g = QGroupBox("프로필")
        gv = QVBoxLayout()
        gv.addWidget(self.lbl_name)
        gv.addWidget(self.lbl_created)
        gv.addWidget(self.lbl_emb_count)
        g.setLayout(gv)

        self.btn_back = QPushButton("뒤로가기")
        self.btn_logout = QPushButton("로그아웃")
        self.btn_delete = QPushButton("회원탈퇴")

        b = QHBoxLayout()
        b.addStretch(1)
        b.addWidget(self.btn_back)
        b.addWidget(self.btn_logout)
        b.addWidget(self.btn_delete)
        b.addStretch(1)

        root = QVBoxLayout(self)
        root.addWidget(self.title)
        root.addWidget(g)
        root.addLayout(b)

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

        # 로그인된 경우 DB 조회
        try:
            with self.ctx.SessionLocal() as s:
                from db.models import User, FaceEmbedding
                user = s.query(User).filter_by(id=self.ctx.current_user_id).one_or_none()
                if not user:
                    # 유저가 사라졌으면 로그아웃 처리
                    self.ctx.clear_current_user()
                    self._show_logged_out_view()
                    return

                emb_count = s.query(FaceEmbedding).filter_by(user_id=user.id).count()

                self.title.setText("내 정보")
                self.lbl_name.setText(f"이름: {user.name}")
                self.lbl_created.setText(f"가입일: {user.created_at}")
                self.lbl_emb_count.setText(f"등록된 얼굴 샘플: {emb_count}개")

                self.btn_logout.setEnabled(True)
                self.btn_delete.setEnabled(True)
        except Exception:
            traceback.print_exc()
            QMessageBox.critical(self, "오류", "정보를 불러오지 못했습니다.")
            self._show_logged_out_view()

    def _show_logged_out_view(self):
        """비로그인 화면"""
        self.title.setText("로그인이 필요합니다")
        self.lbl_name.setText("이름: -")
        self.lbl_created.setText("가입일: -")
        self.lbl_emb_count.setText("등록된 얼굴 샘플: -")

        self.btn_logout.setEnabled(False)
        self.btn_delete.setEnabled(False)

        # 뒤로가기 대신 로그인/처음으로 안내
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
                    s.delete(user)  # ON DELETE CASCADE로 face_embeddings 자동 삭제
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
