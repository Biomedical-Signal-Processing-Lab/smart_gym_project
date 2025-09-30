# core/context.py
import os
from core.camera_manager import CameraManager
from core.face_service import FaceService
from core.config import CAMERA_MAP, WIDTH, HEIGHT, FPS
from db.database import create_engine_and_session, init_db
from core.face_service import FaceService

class AppContext:
    def __init__(self):
        # --- DB 경로 준비 ---
        root = os.path.dirname(os.path.dirname(__file__))  
        data_root = os.path.join(root, "data")
        os.makedirs(data_root, exist_ok=True)

        # --- DB 엔진/세션팩토리 생성 + 테이블 초기화 ---
        db_path = os.path.join(data_root, "app.db")
        self.engine, self.SessionLocal = create_engine_and_session(db_path)
        init_db(self.engine)

        self.face = FaceService(self.SessionLocal)

        self.cam = CameraManager(CAMERA_MAP, WIDTH, HEIGHT, FPS)
        self.router = None
        self.current_exercise = None

        self.current_user_id: int | None = None
        self.current_user_name: str | None = None
        
    def set_router(self, router):
        self.router = router

    def goto_summary(self, summary: dict):
        if not self.router:
            return
        page = self.router.navigate("summary")
        if hasattr(page, "set_data"):
            page.set_data(summary)

    def goto_main(self):
        if self.router:
            self.router.navigate("select")

    def restart_current_exercise(self, ex: str | None):
        if not self.router:
            return
        if ex == "squat":
            self.router.navigate("squat")
        elif ex == "plank":
            self.router.navigate("plank")
        elif ex == "lunge":
            self.router.navigate("lunge")
        elif ex == "pushup":
            self.router.navigate("pushup")
        else:
            self.router.navigate("select")

    def set_current_user(self, user_id: int, name: str):
        self.current_user_id = user_id
        self.current_user_name = name

    def clear_current_user(self):
        self.current_user_id = None
        self.current_user_name = None

    def is_logged_in(self) -> bool:
        return self.current_user_id is not None
