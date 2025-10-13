# rpi_fullscreen_cover.py
import cv2
import numpy as np
import tkinter as tk

def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h

def resize_cover(img, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    scale = max(target_w / w, target_h / h)  # cover: 더 큰 스케일 선택
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    # 중앙 크롭
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]

def main():
    screen_w, screen_h = get_screen_size()
    print(f"Screen: {screen_w}x{screen_h}")

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # /dev/video0
    # 성능 팁: 가능한 MJPG로 받고 내부 디코드
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    win = "CAM"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        # 화면 꽉 채우기(비율 유지, 중앙 크롭)
        frame_fs = resize_cover(frame, screen_w, screen_h)
        cv2.imshow(win, frame_fs)

        # 종료: ESC, 창 전환: F(풀스크린 토글), Q
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        elif k == ord('f'):
            # 토글용: 전체화면 ↔ 창모드
            mode = cv2.getWindowProperty(win, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                win, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL if mode == 1.0 else cv2.WINDOW_FULLSCREEN
            )

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
