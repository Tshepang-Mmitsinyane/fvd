import cv2
from fdf import Detector

cap = cv2.VideoCapture("walking.mp4")


while True:
    ret, frame = cap.read()
    frame = Detector(frame)
    k = cv2.waitKey(30) & 0xff
    if k == "q":
        break

cv2.destroyAllWindows()