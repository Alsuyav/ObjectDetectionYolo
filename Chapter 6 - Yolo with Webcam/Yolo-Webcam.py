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
cap = cv2.VideoCapture("../Videos/bikes.mp4")

model = YOLO("../Yolo-Weights/yolov8l.pt")

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

# prev_frame_time = 0
# new_frame_time = 0

while True:
    # new_frame_time = time.time()

    success, img = cap.read()
    results = model(img, stream=True) # True - более эффективно
    # мы хотим получить ограничивающие прямоугольники для каждого результата
    for r in results:
        # ограничивающие прямоугольники - боксы
        boxes = r.boxes
        for box in boxes:
            # координаты противоположных углов бокса
            x1, y1, x2, y2 = box.xyxy[0]
            # переводим тензоры в int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # для проверки можно распечатать
            # print(x1, y1, x2, y2)

            # нарисуем этот прямоугольник, чтобы его увидеть

            # первый вариант
            # задаем координаты, цвет и толщину
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # второй вариант
            w, h = x2 - x1, y2 - y1
            # задаем координаты левого верхнего угла, ширину, высоту, цвета
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255,0,255), colorC=(255,0,255))

            # уверенность
            conf = math.ceil((box.conf[0] * 100)) / 100
            # имя класса, конкретнее - его номер в списке классов
            cls = int(box.cls[0])

            # помещение над боксом и центрирование текста(имя класса и уверенность) в зависимости от его размера
            # (max(0, x1), max(35, y1)) - сделано для того, чтобы текст не выходил за рамки видимости
            # scale=1 - контролирует размер текста, например, 0.5 сделает текст меньше
            # thickness=1 - контролирует толщину
            cvzone.putTextRect(img, f' {classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # fps = 1 / (new_frame_time - prev_frame_time)
    # prev_frame_time = new_frame_time
    # print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)