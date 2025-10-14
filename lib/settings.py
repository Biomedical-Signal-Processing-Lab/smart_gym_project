# settings.py
import os

# 모델/포스트프로세스 경로 (env로도 덮어쓰기 가능)
HEF       = os.environ.get("HEF", "/home/pi/workspace/hailo/yolov8s_pose.hef")
POST_SO   = os.environ.get("POST_SO", "/usr/local/hailo/resources/so/libyolov8pose_postprocess.so")
POST_FUNC = os.environ.get("POST_FUNC", "filter_letterbox")
CROPPER_SO= os.environ.get("CROPPER_SO", "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/cropping_algorithms/libwhole_buffer.so")

# 입력 소스(카메라)
CAM = os.environ.get("CAM", "/dev/video0")
SRC_WIDTH  = int(os.environ.get("SRC_WIDTH",  "1280"))
SRC_HEIGHT = int(os.environ.get("SRC_HEIGHT", "720"))
SRC_FPS    = int(os.environ.get("SRC_FPS",    "30"))

# 창/표시
WINDOW_TITLE = "Hailo YOLOv8-Pose (overlay + appsink)"
FULLSCREEN   = True
WINDOW_SIZE  = (1280, 720)
SHOW_INFO_BAR= True



