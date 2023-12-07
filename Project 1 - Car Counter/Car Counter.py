"""
Чтобы считать машины, нужно определить некоторую область и детектировать машины в ней.
Для этого необходимо создать маску (можно с помощью canva) - mask.png
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
cap = cv2.VideoCapture("../Videos/cars.mp4")

model = YOLO("../Yolo-Weights/yolov8n.pt")

# список того, что будем идентифицировать - список классов
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

# пределы линии, которую будут пересекать машины
limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    # наложение побитовой маски
    imgRegion = cv2.bitwise_and(img, mask)

    # добавим картинку для красивого представления подсчётов
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

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

            # хотим определять только 4 перечисленных класса с уверенностью более 0.3
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # выводит бокс и надпись для всех машин
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                # сохраняем в список детекций
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # трекер обновляется списком детекций
    resultsTracker = tracker.update(detections)

    # отображение линии, будут учитываться машины пересекающие эту линию
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

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
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # учитываем только уникальный id, т.е. которых еще нет в списке
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # линия будет зеленой, когда идентифицирует объект
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # простой счётчик в левом верхнем углу
        # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))

        # красивый счётчик
        cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    # показывает только выделенный участок
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
