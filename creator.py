import cv2
import time
import torch
from ultralytics import YOLO
import os

# 📌 URL потока
stream_url = "http://46.191.199.17/1557318547RKB341/index.fmp4.m3u8?token=0cc12cd37d3a44129932b9d201928fdf"

# 📌 Проверка на GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Используем устройство: {device.upper()}")

# 📌 Загрузка модели YOLO
model = YOLO("best_x3.pt").to(device)

# 📌 Открытие видеопотока
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ Ошибка: OpenCV не может открыть поток!")
    exit()

# 📌 Папки для сохранения
output_images = "training_data/images10/"
output_labels = "training_data/labels10/"
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# ⚙ Параметры сохранения
frame_interval = 5             # Секунд между кадрами
max_frames = 100               # Максимум сохранённых кадров
saved_frame_count = 0          # Счётчик сохранённых кадров
last_saved_time = 0            # Последнее сохранённое время

while cap.isOpened() and saved_frame_count < max_frames:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Ошибка: не удалось получить кадр.")
        break

    current_time = time.time()

    if current_time - last_saved_time >= frame_interval:
        results = model(frame)
        timestamp = int(current_time)
        image_filename = f"{output_images}{timestamp}.jpg"
        label_filename = f"{output_labels}{timestamp}.txt"

        with open(label_filename, "w") as label_file:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                x_center, y_center, width, height = box.xywhn[0]
                label_file.write(f"{cls} {x_center} {y_center} {width} {height}\n")

        cv2.imwrite(image_filename, frame)
        print(f"✅ Сохранены: {image_filename} и {label_filename}")
        last_saved_time = current_time
        saved_frame_count += 1

    cv2.imshow("YOLO Stream (GPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
