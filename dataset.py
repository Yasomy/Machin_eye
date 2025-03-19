import cv2
import time
import torch
from ultralytics import YOLO
import os

# 📌 URL потока (замените на актуальный)
stream_url = "http://46.191.199.9/1660720512DSH176/tracks-v1/index.fmp4.m3u8?token=0787772081c54d76b3283638320c30b3"

# 📌 Проверяем, есть ли GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Используем устройство: {device.upper()}")

# 📌 Загружаем YOLO (с моделью 12x)
model = YOLO("yolo12x.pt").to(device)

# 📌 Открываем поток
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ Ошибка: OpenCV не может открыть поток!")
    exit()

# 📌 Настройка сохранения данных
output_images = "training_data/images/"
output_labels = "training_data/labels/"
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

frame_interval = 1  # Ограничение - 1 кадр в секунду
last_saved_time = 0  # Время последнего сохраненного кадра

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Ошибка: не удалось получить кадр.")
        break

    # 📌 Текущее время
    current_time = time.time()

    # 📌 YOLO делает предсказание
    results = model(frame)

    # 📌 Файл для сохранения
    timestamp = int(current_time)
    image_filename = f"{output_images}{timestamp}.jpg"
    label_filename = f"{output_labels}{timestamp}.txt"

    # 📌 Открываем файл для разметки
    with open(label_filename, "w") as label_file:
        for box in results[0].boxes:
            cls = int(box.cls[0])  # Класс объекта
            conf = float(box.conf[0])  # Доверие модели
            x_center, y_center, width, height = box.xywhn[0]  # Нормализованные координаты YOLO

            # 📌 Записываем в .txt файл в формате YOLO (class x_center y_center width height)
            label_file.write(f"{cls} {x_center} {y_center} {width} {height}\n")

    # 📌 Сохранение изображения
    if current_time - last_saved_time >= frame_interval:
        cv2.imwrite(image_filename, frame)
        print(f"✅ Сохранены: {image_filename} и {label_filename}")

        # Обновляем время последнего сохраненного кадра
        last_saved_time = current_time

    # 📌 Показываем обработанный кадр
    cv2.imshow("YOLO Stream (GPU)", frame)

    # 📌 Выход по "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
