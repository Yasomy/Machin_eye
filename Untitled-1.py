import cv2
from ultralytics import YOLO

# Вставьте ваш `m3u8` URL в кавычках!
stream_url = "http://46.191.199.16/1659331721AYT463/tracks-v1/index.fmp4.m3u8?token=b56933a810d64b3e96b9c04f87700def"

# Загружаем модель YOLOv8
model = YOLO("yolov8s.pt")  

# Открываем поток
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌ Ошибка: OpenCV не может открыть поток!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Ошибка: не удалось получить кадр.")
        break

    # Запускаем YOLO
    results = model(frame)
    frame = results[0].plot()  # Отрисовываем детекции

    cv2.imshow("YOLO Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Нажмите "q" для выхода
        break

cap.release()
cv2.destroyAllWindows()
