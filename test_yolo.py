from ultralytics import YOLO

print("🔹 Загружаю модель YOLOv8...")

# Загружаем предобученную модель YOLOv8
model = YOLO("yolov8n.pt")

print("🔹 Делаю предсказание...")

# Тест на изображении Zidane (из интернета)
results = model("https://ultralytics.com/images/zidane.jpg")

print("🔹 Готово, показываю изображение!")
results[0].show()

print("✅ Скрипт завершён")
