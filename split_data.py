from ultralytics import YOLO
import os

# 📌 Проверяем файлы разметки
labels_dir = "D:/Tracking_Cam/training_data/labels"
num_empty_files = 0

for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    if os.path.getsize(label_path) == 0:  # Если файл пустой
        os.remove(label_path)
        num_empty_files += 1

print(f"✅ Удалено пустых файлов: {num_empty_files}")

if __name__ == "__main__":
    # 📌 Загружаем модель YOLO
    model = YOLO("yolo12x.pt")

    # 📌 Запускаем обучение
    model.train(
        data="D:/Tracking_Cam/training_data/data.yaml",
        epochs=50,      # Количество эпох
        batch=16,       # Размер batch
        imgsz=640,      # Размер изображений
        device="cuda"   # Используем GPU
    )
