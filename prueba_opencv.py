from ultralytics import YOLO
import cv2

captura = cv2.VideoCapture('Videos/Video1.mp4')
while (captura.isOpened()):
    ret, imagen = captura.read()
    if ret == True:
        cv2.imshow('video', imagen)
    if cv2.waitKey(30) == ord('s'):
        break
    else: break
captura.release()
cv2.destroyAllWindows()