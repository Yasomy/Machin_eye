import cv2
import time
from ultralytics import YOLO

# URL потока
stream_url = "http://46.191.199.16/1659331721AYT463/tracks-v1/index.fmp4.m3u8?token=b56933a810d64b3e96b9c04f87700def"

# Загружаем YOLO
model = YOLO("yolov8s.pt")  

# Открываем поток
cap = cv2.VideoCapture(stream_url)

# Настройка записи видео
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))

# Фиксация времени для FPS
prev_time = 0
fps_limit = 10  # Ограничение кадров в секунду

while cap.isOpened():
    curr_time = time.time()
    elapsed_time = curr_time - prev_time

    if elapsed_time < 1.0 / fps_limit:
        continue  # Пропускаем кадры для ограничения FPS

    ret, frame = cap.read()
    if not ret:
        print("❌ Ошибка: не удалось получить кадр.")
        break

    # YOLO делает предсказание
    results = model(frame)
    
    # Объединяем транспорт в один класс
    transport_classes = {2, 3, 5, 7}  # (автомобиль, мотоцикл, автобус, грузовик)
    person_count = 0
    transport_count = 0

    for box in results[0].boxes:
        cls = int(box.cls[0])  # Класс объекта
        if cls == 0:  # Человек
            person_count += 1
        elif cls in transport_classes:  # Транспорт
            transport_count += 1

    # Отображаем FPS
    fps = 1 / elapsed_time
    prev_time = curr_time

    # Добавляем текст
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"People: {person_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Transport: {transport_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Отрисовка YOLO
    frame = results[0].plot()

    # Запись видео
    out.write(frame)

    # Отображение окна
    cv2.imshow("YOLO Stream", frame)

    # Выход по "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
