"""
Также необходим трекер для каждого авто, чтобы не принимать одну и ту же машину на разных кадрах за разные объекты.
Используем sort.py - https://github.com/abewley/sort
"""


import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


# создаем объект для видео
cap = cv2.VideoCapture("../Videos/people.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

# список классов
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# наша маска
mask = cv2.imread("mask.png")

# создаем объект трекера
# max_age - максимальное кол-во кадров для ожидания потерянного объекта
# min_hits - минимальное количество попаданий
# iou_threshold - подог пересечение / объединение
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# пределы линий, которые будут пересекаться людьми, едущими вверх и вниз
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    # наложение побитовой маски
    imgRegion = cv2.bitwise_and(img, mask)

    # добавим картинку для красивого представления подсчётов и её местоположение
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (730, 260))

    results = model(imgRegion, stream=True)

    # список детекций
    # (0, 5) - этот формат указан в sort
    detections = np.empty((0, 5))

    # мы хотим получить ограничивающие прямоугольники для каждого результата
    for r in results:
        # ограничивающие прямоугольники - боксы
        boxes = r.boxes
        for box in boxes:
            # координаты противоположных углов бокса
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # уверенность
            conf = math.ceil((box.conf[0] * 100)) / 100
            # номер класса в списке классов
            cls = int(box.cls[0])
            # имя класса
            currentClass = classNames[cls]

            # хотим определять только 1 класс с уверенностью более 0.3
            if currentClass == "person" and conf > 0.3:
                # выводит бокс и надпись для всех людей
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                # сохраняем в список детекций
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # трекер обновляется списком детекций
    resultsTracker = tracker.update(detections)

    # отображение линий, будут учитываться люди пересекающие эту линию
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    # просмотр результатов трекера
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1

        # отображение бокса
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        # добавление точки центр объекта
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # если центр объект оказывается около линии, засчитываем объект
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

        # счётчик
        cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
        cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
