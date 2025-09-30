FRONT = "/dev/v4l/by-id/usb-Generic_USB2.0_PC_CAMERA-video-index0"  # = /dev/video0
SIDE  = "/dev/v4l/by-id/usb-Wed_Camera_Wed_Camera_20240420112206-video-index0"  # = /dev/video2

CAMERA_MAP = {
    "front": FRONT,
    "side":  SIDE,
}

WIDTH, HEIGHT, FPS = 640, 480, 30
