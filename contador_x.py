from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import math
from sort import Sort

# Configuración de captura de video
cap1 = cv2.VideoCapture("Videos/Video1.mp4")
cap2 = cv2.VideoCapture("Videos/Video2.mp4")

model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Seguimiento/rastreador
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Límites para la línea de conteo
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

# Resolución de destino
target_width = 1280
target_height = 720

def process_frame(frame, model, tracker):
    results = model(frame, stream=True)
    detections = np.empty((0, 5))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0] * 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 30:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
    
    resultsTracker = tracker.update(detections)
    
    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(frame, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 20 < cy < limitsUp[1] + 20:
            if totalCountUp.count(Id) == 0:
                totalCountUp.append(Id)
                cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 3)
        
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 20 < cy < limitsDown[1] + 20:
            if totalCountDown.count(Id) == 0:
                totalCountDown.append(Id)
                cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 3)
    
    cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 3)
    cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 3)
    
    cv2.putText(frame, str(len(totalCountUp)), (1050, 345), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 7)
    cv2.putText(frame, str(len(totalCountDown)), (1050, 450), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
    
    return frame

while True:
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    
    if not success1 or not success2:
        break
    
    processed_frame1 = process_frame(frame1, model, tracker)
    processed_frame2 = process_frame(frame2, model, tracker)
    
    combined_frame = np.hstack((processed_frame1, processed_frame2))
    combined_frame = cv2.resize(combined_frame, (target_width, target_height))
    
    cv2.imshow("Combined Videos", combined_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
