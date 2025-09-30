# core/context.py
from core.camera_manager import CameraManager
from core.face_service import FaceService
import os
from core.config import CAMERA_MAP, WIDTH, HEIGHT, FPS

class AppContext:
    def __init__(self):
        self.cam = CameraManager(CAMERA_MAP, WIDTH, HEIGHT, FPS)
        self.router = None
        self.current_exercise = None

        root = os.path.dirname(os.path.dirname(__file__))  
        data_dir = os.path.join(root, "data", "faces")
        os.makedirs(data_dir, exist_ok=True)
        self.face = FaceService(store_dir=data_dir)

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
            self.router.navigate("start")

    def restart_current_exercise(self, ex: str | None):
        if not self.router:
            return
        if ex == "squat":
            self.router.navigate("squat")
        elif ex == "plank":
            self.router.navigate("plank")
        else:
            self.router.navigate("select")
