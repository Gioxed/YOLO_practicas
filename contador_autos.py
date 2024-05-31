from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("Videos/VideoAutos.mp4") #para video
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#mascara para detectar solamente esa zona
mask = cv2.imread("Imagenes/mask.png")

#seguimiento
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    sucess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    
    results = model(imgRegion, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            #cuadro delimitador
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255,),1)
            w, h = x2-x1, y2- y1 #calcula el ancho y el alto
            
            #confianza redondeada con math
            conf = box.conf[0] * 100
            
            #Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            #solo muestra los dichos y si el nivel de confianza es bajo tampoco lo muestra
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 30:
                #mostrar un rectangulo con la confianza arriba de la caja sin pasarse del borde
                cvzone.putTextRect(img, f"{classNames[cls]} {conf:.2f}%", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3) #scale = hacer el tamaño de la fuente mas chica, thickness = espesor, offset = tamaño del cuadro
                cvzone.cornerRect(img, (x1, y1, w, h), l=9) #L = espacio entre los bordes
            
    cv2.imshow("Imagenes", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

