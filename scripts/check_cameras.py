import sys

import cv2

# On the Pi, USB webcams appear after the CSI/ISP nodes (e.g. /dev/video8)
for i in range(16):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camera {i}: FOUND - {int(cap.get(3))}x{int(cap.get(4))}")
        cap.release()
    else:
        print(f"Camera {i}: not found")
