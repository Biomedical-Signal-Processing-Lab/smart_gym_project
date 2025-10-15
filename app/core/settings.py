# core/settings.py
import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_ROOT / "models"

CAM = os.environ.get(
    "CAM",
    "/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM_JH1901_20240311_v007-video-index0"
)

SRC_WIDTH  = int(os.environ.get("SRC_WIDTH",  "1280"))
SRC_HEIGHT = int(os.environ.get("SRC_HEIGHT", "720"))
SRC_FPS    = int(os.environ.get("SRC_FPS",    "30"))

HEF      = os.environ.get("HEF", str(MODELS_DIR / "yolov8s_pose.hef"))
POST_SO  = str(MODELS_DIR / "libyolov8pose_postprocess.so")
POST_FUNC= os.environ.get("POST_FUNC", "filter_letterbox")

CROPPER_SO = os.environ.get(
    "CROPPER_SO",
    "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so"
)

WINDOW_TITLE = "Hailo YOLOv8-Pose"
FULLSCREEN   = False
WINDOW_SIZE  = (SRC_WIDTH, SRC_HEIGHT)
SHOW_INFO_BAR = False

