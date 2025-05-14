import cv2
import time
import os

# 📌 URL потока
stream_url = "http://46.191.199.34/001-999-4-348/tracks-v1/index.fmp4.m3u8?token=7447c14f01e0486ba2293cbf9d45128d"

# 📌 Открываем поток
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("❌ Ошибка: не удалось открыть поток")
    exit()

# 📁 Папка для сохранения скриншотов
output_dir = "screenshots"
os.makedirs(output_dir, exist_ok=True)

# ⏱️ Настройка времени
interval = 5  # секунд между скриншотами
max_shots = 47
last_time = time.time()
shot_count = 0

print("📸 Начинаем запись скриншотов...")

while cap.isOpened() and shot_count < max_shots:
    ret, frame = cap.read()
    if not ret:
        print("❌ Ошибка: не удалось считать кадр")
        break

    current_time = time.time()
    if current_time - last_time >= interval:
        filename = os.path.join(output_dir, f"{int(current_time)}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Сохранён: {filename}")
        last_time = current_time
        shot_count += 1

    # Можно показывать кадры, если нужно:
    # cv2.imshow("Stream", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
print("🎉 Готово! Скриншоты сохранены.")
