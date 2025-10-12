# display.py
import cv2

def make_window(title: str, fullscreen: bool, size: tuple[int,int]):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        w, h = size
        cv2.resizeWindow(title, w, h)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

def put_info_bar(frame, text: str, show: bool=True):
    if not show:
        return
    h, w = frame.shape[:2]
    bar_h = 28
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2, cv2.LINE_AA)
