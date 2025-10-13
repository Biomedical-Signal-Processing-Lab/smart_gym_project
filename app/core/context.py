# core/context.py
import os
from core.camera_manager import CameraManager
from core.face_service import FaceService
from core.config import CAMERA_DEVICE, WIDTH, HEIGHT, FPS
from db.database import create_engine_and_session, init_db
from core.face_service import FaceService
from db.models import WorkoutSession

class AppContext:
    def __init__(self):
        root = os.path.dirname(os.path.dirname(__file__))  
        data_root = os.path.join(root, "data")
        os.makedirs(data_root, exist_ok=True)

        db_path = os.path.join(data_root, "app.db")
        self.engine, self.SessionLocal = create_engine_and_session(db_path)
        init_db(self.engine)

        self.face = FaceService(self.SessionLocal)

        self.cam = CameraManager(CAMERA_DEVICE, WIDTH, HEIGHT, FPS)
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

    def goto_profile(self):
        if self.router:
            self.router.navigate("info")

    def restart_current_exercise(self, ex: str | None = None):
        if self.router:
            self.router.navigate("exercise") 

    def set_current_user(self, user_id: int, name: str):
        self.current_user_id = user_id
        self.current_user_name = name

    def clear_current_user(self):
        self.current_user_id = None
        self.current_user_name = None

    def is_logged_in(self) -> bool:
        return self.current_user_id is not None
    
    def save_workout_session(self, summary: dict):
        if not self.is_logged_in():
            return

        from datetime import datetime
        started = summary.get("started_at")
        ended   = summary.get("ended_at")

        with self.SessionLocal() as s:
            sess = WorkoutSession(
                user_id=self.current_user_id,
                exercise=summary.get("exercise", "unknown"),
                reps=int(summary.get("reps", 0)),
                avg_score=float(summary.get("avg_score", 0.0)),
                duration_sec=int(summary.get("duration_sec", 0)),
                started_at=datetime.fromtimestamp(started) if started else None,
                ended_at=datetime.fromtimestamp(ended) if ended else None,
            )
            s.add(sess)
            s.commit()
