from ultralytics import YOLO
import cv2
import cvzone
import math
# import time

# создаем объект веб-камеры
# нужно раскомментировать, чтобы использовать свою веб-камеру
# cap = cv2.VideoCapture(0) # 0 - т.к. одна вебка, 1 - две и.т.д.
# cap.set(3, 1280) # высота
# cap.set(4, 720) # ширина

# создаем объект для видео
# использование загруженных видео (в каталоге Videos) для детекции
# нужно закомментировать, чтобы использовать свою веб-камеру
cap = cv2.VideoCapture("../Videos/ppe-3.mp4")

# модель, предварительно обученная в Google Colab
# на датасете - https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/27
model = YOLO("ppe.pt")

# список того, что будем идентифицировать - список классов
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    # мы хотим получить ограничивающие прямоугольники для каждого результата
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # бокс
            # координаты противоположных углов бокса
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # отобразить бокс
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # уверенность
            conf = math.ceil((box.conf[0] * 100)) / 100
            # имя класса
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            # цвет бокса будет зависеть от класса
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)