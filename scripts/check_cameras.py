import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f"Camera {i}: FOUND - {int(cap.get(3))}x{int(cap.get(4))}")
        cap.release()
    else:
        print(f"Camera {i}: not found")
