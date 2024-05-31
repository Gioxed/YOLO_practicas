#Hacer que cuenta cuantas personas suben y bajan la escalera
from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import Sort

cap = cv2.VideoCapture("Videos/people.mp4") #para video
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#mascara para detectar solamente esa zona
mask = cv2.imread("Imagenes/mask2.png")

#seguimiento/rastreador
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#limites para crear una linea y cada vez que pasa de ese punto me cuenta el objeto
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

while True:
    sucess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    
    imgGraphics = cv2.imread("Imagenes/graphics.png", cv2.IMREAD_UNCHANGED)
    
    # Verificar si la imagen tiene 3 canales (RGB) o 4 canales (RGBA)
    if imgGraphics.shape[2] == 3:  # Si tiene 3 canales (RGB), agregar un canal alfa
        b, g, r = cv2.split(imgGraphics)
    # Crear un canal alfa completamente opaco (255) del mismo tamaño que la imagen
        alpha = 255 * np.ones_like(b, dtype=b.dtype)
    # Fusionar los canales RGB con el canal alfa para formar una imagen RGBA
        imgGraphics = cv2.merge((b, g, r, alpha))
    
    imgGraphics = cv2.resize(imgGraphics, (0, 0), fx=0.5, fy=0.5)
    
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))
    results = model(imgRegion, stream=True)
    
    detections = np.empty((0, 5))
    
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
            if currentClass == "person" and conf > 30:
                #mostrar un rectangulo con la confianza arriba de la caja sin pasarse del borde
                #cvzone.putTextRect(img, f"{classNames[cls]} {conf:.2f}%", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3) #scale = hacer el tamaño de la fuente mas chica, thickness = espesor, offset = tamaño del cuadro
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=3) #L = espacio entre los bordes, rt = espesor del rectangulo
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
            
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 3) #,5 espesor
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 3)
    #identificador
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2-x1, y2- y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        
        #realizar el conteo a partir de la linea
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx, cy), 5, (255,0,255), cv2.FILLED) #cuando el ciculo cae en la región, realiza el conteo
        
    #     if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 0:
    #         if totalCount.count(Id) == 0:
    #             totalCount.append(Id) #contar el numero por la id
    #             cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
            
    # #cvzone.putTextRect(img, f"Contador:{len(totalCount)}", (50, 50))
    # cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 8)
            
    cv2.imshow("Imagenes", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
    