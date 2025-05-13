import cv2
import time
import torch
from ultralytics import YOLO

# URL потока
stream_url = "http://46.191.199.12/1703147374/tracks-v1/index.fmp4.m3u8?token=1ce8cb84810341a48f657e49bf7ed565"

# Проверяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Используем устройство: {device.upper()}")

# Загружаем модель
model = YOLO("best_x3.pt").to(device)

# Открываем видеопоток
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ Ошибка: OpenCV не может открыть поток!")
    exit()

# Получаем размеры видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if width == 0 or height == 0:
    print("❌ Ошибка: неверный размер видео!")
    exit()

# Настройки
input_size = 1920
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (width, height))
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Ошибка: не удалось получить кадр.")
        break

    # Изменение размера для YOLO
    resized_frame = cv2.resize(frame, (input_size, input_size))

    # Предсказание YOLO
    results = model(resized_frame)

    # Счётчики и фильтрация классов
    person_count = 0
    transport_count = 0
    filtered_boxes = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:
            person_count += 1
            filtered_boxes.append(box)
        elif cls == 2:
            transport_count += 1
            filtered_boxes.append(box)

    results[0].boxes = filtered_boxes

    # Отрисовка результатов с уменьшенной толщиной рамки
    frame = results[0].plot(line_width=1)
    frame = cv2.resize(frame, (width, height))

    # Расчёт FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Добавление текста
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"People: {person_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Transport: {transport_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    out.write(frame)
    cv2.imshow("YOLO Stream (GPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
