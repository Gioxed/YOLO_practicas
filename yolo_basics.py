from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
results = model("Imagenes/Autos.jpg", show=True)
cv2.waitKey(0)