#Hacer que cuenta cuantas personas suben y bajan la escalera
from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("Videos/Video2.mp4")


while True:
    success, img = cap.read()
    results = model(img, show=True)

cv2.imshow("Imagenes", img)

cv2.waitKey(0)
cap.release()

