# core/context.py
from dataclasses import dataclass, field
from typing import Optional
from camera_manager import CameraManager
from config import CAMERA_MAP, WIDTH, HEIGHT, FPS

@dataclass
class AppContext:
    cam: CameraManager = field(default_factory=lambda: CameraManager(CAMERA_MAP, WIDTH, HEIGHT, FPS))
    # 여기에 전역 설정, 프로파일, 이벤트 버스 등을 추가
