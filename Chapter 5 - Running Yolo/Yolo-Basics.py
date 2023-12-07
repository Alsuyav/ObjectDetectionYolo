from ultralytics import YOLO
import cv2


# model = YOLO('yolov8n.pt')
# YOLO version 8, n - nano (тип весов), m - medium, l - large
# - появился файл yolov8n.pt,
# его переместим в папку Yolo-Weights, специально созданной, это всё нужно, чтобы не скачивать веса каждый раз

model = YOLO('../Yolo-Weights/yolov8m.pt')

# передаем источник изображение и указываем, что хотим увидеть изображение в конце
results = model("Images/3.png", show=True)
# благодаря этой строчки выходное изображение не закрывается мгновенно
cv2.waitKey(0)


# При запуске всего файла получаем:
# Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt to yolov8n.pt...
# 100%|██████████| 6.23M/6.23M [00:04<00:00, 1.48MB/s]
# Ultralytics YOLOv8.0.26  Python-3.10.9 torch-2.1.1+cpu CPU
# YOLOv8n summary (fused): 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
#
# image 1/1 C:\Users\alsuy\PycharmProjects\ObjectDetectionYolo\Chapter 5 - Running Yolo\Images\1.png: 384x640 6 persons, 2 buss, 2 backpacks, 1 handbag, 210.0ms
# Speed: 2.0ms pre-process, 210.0ms inference, 20.0ms postprocess per image at shape (1, 3, 640, 640)
# тут указаны скорости и размер изображения (1, каналы, ширина, высота)



