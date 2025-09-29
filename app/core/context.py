# core/context.py
from camera_manager import CameraManager
from config import CAMERA_MAP, WIDTH, HEIGHT, FPS

class AppContext:
    def __init__(self):
        self.cam = CameraManager(CAMERA_MAP, WIDTH, HEIGHT, FPS)
        self.router = None
        self.current_exercise = None

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
        else:
            self.router.navigate("select")
