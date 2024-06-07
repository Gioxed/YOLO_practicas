from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import Sort

# Configuración de captura de video
cap = cv2.VideoCapture("Videos/Video1.mp4")
#cap = cv2.VideoCapture("Videos/Video2.mp4")

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Seguimiento/rastreador
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

#limites para crear una linea y cada vez que pasa de ese punto me cuenta el objeto
limitsUp = [527, 540, 735, 540]
limitsDown = [527, 500, 735, 500]

#registrar las IDs de las personas que han cruzado las líneas de conteo en cada dirección. Esto evita que una persona sea contada dos veces si cruza ambas líneas.
totalCountUp = []
totalCountDown = []

# Diccionarios para registrar cruces de lineas
crossed_up = {}
crossed_down = {}

# Resolución de destino
target_width = 1280
target_height = 720

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Redimensionar el video manteniendo la relación de aspecto
    height, width = img.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cv2.resize(img, (new_width, new_height))
    
    # Crear una imagen negra de fondo
    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calcular las coordenadas para centrar el video redimensionado
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Colocar el video redimensionado en el fondo negro
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
    
    img = background
    
    imgGraphics = cv2.imread("Imagenes/graphics.png", cv2.IMREAD_UNCHANGED)
    
    # Verificar si la imagen tiene 3 canales (RGB) o 4 canales (RGBA)
    if imgGraphics.shape[2] == 3:  # Si tiene 3 canales (RGB), agregar un canal alfa
        b, g, r = cv2.split(imgGraphics)
    # Crear un canal alfa completamente opaco (255) del mismo tamaño que la imagen
        alpha = 255 * np.ones_like(b, dtype=b.dtype)
    # Fusionar los canales RGB con el canal alfa para formar una imagen RGBA
        imgGraphics = cv2.merge((b, g, r, alpha))
    
    imgGraphics = cv2.resize(imgGraphics, (0, 0), fx=0.5, fy=0.5)
    
    #ubicacion de la imagen
    img = cvzone.overlayPNG(img, imgGraphics, (900, 260))
    results = model(img, stream=True)
    
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
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 3) #,3 espesor
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
        
        #Antes de incrementar el contador, se verifica si la ID no está en crossed_up ni en crossed_down. Si no lo está, se añade la ID a crossed_up para que no sea contada de nuevo si cruza la línea de bajada. , y viceversa
        #contador personas subiendo
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 20 < cy < limitsUp[1] + 0:
            if Id not in crossed_up and Id not in crossed_down:
                totalCountUp.append(Id) #contar el numero por la id
                crossed_up[Id] = True
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 3)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 3)
            
        #contador personas bajando
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 20 < cy < limitsDown[1] + 0:
            if Id not in crossed_down and Id not in crossed_up:
                totalCountDown.append(Id) #contar el numero por la id
                crossed_down[Id] = True
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 3)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 3)
                
                
    # #cvzone.putTextRect(img, f"Contador:{len(totalCount)}", (50, 50))
    #ubicacion del contador
    cv2.putText(img, str(len(totalCountUp)), (1150, 345), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 7)
    cv2.putText(img, str(len(totalCountDown)), (1150, 450), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
            
    cv2.imshow("Imagenes", img)
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
    