import cv2
import time
from ultralytics import YOLO

# URL потока (замените на актуальный m3u8, если изменился)
stream_url = "http://46.191.199.9/1660720512DSH176/tracks-v1/index.fmp4.m3u8?token=59b1b67bea794177b3f90864427e228b"

# Загружаем YOLO
model = YOLO("yolo11n.pt")

# Открываем поток
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Ошибка: OpenCV не может открыть поток!")
    exit()

# Получаем размеры видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Проверка размера кадра
if width == 0 or height == 0:
    print("❌ Ошибка: неверный размер видео!")
    exit()

# Настройка записи видео
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (width, height))

prev_time = 0  # Время для вычисления FPS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Ошибка: не удалось получить кадр.")
        break

    # YOLO делает предсказание
    results = model(frame)

    # Объединяем транспорт в один класс
    transport_classes = {2}  # (автомобиль, мотоцикл, автобус, грузовик)
    person_count = 0
    transport_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == 0:  # Человек
            person_count += 1
        elif cls in transport_classes:  # Транспорт
            transport_count += 1

    # Вычисляем FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Добавляем текстовую информацию
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"People: {person_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Transport: {transport_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Отрисовка предсказаний YOLO
    frame = results[0].plot()

    # Запись видео
    out.write(frame)

    # Показываем обработанный кадр
    cv2.imshow("YOLO Stream", frame)

    # Выход по "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
