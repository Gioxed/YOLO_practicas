from ultralytics import YOLO
import cv2
import cvzone
import math

#cap = cv2.VideoCapture("0") #para usar en webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("Videos/Video2.mp4") #para video

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

while True:
    sucess, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            #cuadro delimitador
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255,),3)
            w, h = x2-x1, y2- y1 #calcula el ancho y el alto
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            #confianza redondeada con math
            conf = math.ceil((box.conf[0] * 100)/100)
            
            #Class Name
            clase = int(box.cls[0])
            #mostrar un rectangulo con la confianza arriba de la caja sin pasarse del borde
            cvzone.putTextRect(img, f"{classNames[clase]} {conf}", (max(0, x1), max(35, y1)), scale=0.7, thickness=1) #scale = hacer el tamaño de la fuente mas chica, thickness = espesor
            
            
    cv2.imshow("Imagenes", img)
    cv2.waitKey(1)

