import cv2
import time
import torch
import os
from ultralytics import YOLO

# URL потока (замените на актуальный `m3u8`, если изменился)
stream_url = "http://46.191.199.9/1660720512DSH176/tracks-v1/index.fmp4.m3u8?token=3e32d84f2f7a4650a7bf9a54c9745605"

# Проверяем, есть ли GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Используем устройство: {device.upper()}")

# Загружаем YOLO на GPU (если доступен)
model = YOLO("yolo11n.pt").to(device)

# Открываем поток
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Ошибка: OpenCV не может открыть поток!")
    exit()

# Создаем папку для сохранения скриншотов
output_dir = "dataset_screenshots"
os.makedirs(output_dir, exist_ok=True)

prev_time = 0  # Время для вычисления FPS
frame_count = 0  # Счетчик кадров

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Ошибка: не удалось получить кадр.")
        break

    frame_count += 1

    # YOLO делает предсказание на GPU
    results = model(frame)

    # Фильтрация классов (оставляем только "Person" и "Car")
    person_count = 0
    transport_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты бокса
        conf = box.conf[0]  # Уверенность

        if cls == 0:  # Человек
            person_count += 1
            color = (0, 255, 0)  # Зеленый
            label = f"Person {conf:.2f}"
        elif cls == 2:  # Машина
            transport_count += 1
            color = (255, 0, 0)  # Синий
            label = f"Car {conf:.2f}"
        else:
            continue  # Пропускаем остальные классы

        # Рисуем бокс и подпись
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Если на кадре есть люди или машины, делаем скриншот
    if person_count > 0 or transport_count > 0:
        screenshot_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"📸 Сохранен скриншот: {screenshot_path}")

    # Вычисляем FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Добавляем текстовую информацию
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"People: {person_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Transport: {transport_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Показываем обработанный кадр
    cv2.imshow("YOLO Stream (GPU)", frame)

    # Выход по "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
