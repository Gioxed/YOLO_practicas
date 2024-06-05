from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import Sort

# def mouse_event(event, x, y, flags, param):
#     if event == cv2.EVENT_MOUSEMOVE:
#         print(f"Coordenadas: ({x}, {y})")
        
# # Crear una ventana
# cv2.namedWindow("Video")

# # Asociar la función de eventos del mouse con la ventana
# cv2.setMouseCallback("Video", mouse_event)

cap = cv2.VideoCapture("Videos/cargadora3.mp4") #para video
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#mascara para detectar solamente esa zona
mask = cv2.imread("Imagenes/mask3.png")

#seguimiento/rastreador
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#limites para crear una linea y cada vez que pasa de ese punto me cuenta el objeto
#limits = [400, 100, 400, 350]
limits = [150, 200, 550, 200]

totalCount = []

while True:
    sucess, img = cap.read()
    
    # Redimensionar la máscara para que tenga el mismo tamaño que la imagen
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img, mask)
    
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
            if currentClass == "truck" and conf > 30:
                #mostrar un rectangulo con la confianza arriba de la caja sin pasarse del borde
                #cvzone.putTextRect(img, f"{classNames[cls]} {conf:.2f}%", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3) #scale = hacer el tamaño de la fuente mas chica, thickness = espesor, offset = tamaño del cuadro
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=3) #L = espacio entre los bordes, rt = espesor del rectangulo
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
            
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3) #,5 espesor
    
    #identificador
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        
        #realizar el conteo a partir de la linea
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx, cy), 5, (255,0,255), cv2.FILLED) #cuando el ciculo cae en la región, realiza el conteo
        
        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 0:
            if totalCount.count(Id) == 0:
                totalCount.append(Id) #contar el numero por la id
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
            
    cvzone.putTextRect(img, f"Contador:{len(totalCount)}", (50, 50))
            
    cv2.imshow("Imagenes", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)