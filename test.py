from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
import time

# Включаем логирование запросов
capabilities = DesiredCapabilities.CHROME
capabilities["goog:loggingPrefs"] = {"performance": "ALL"}

# Запускаем Edge в режиме DevTools
options = webdriver.EdgeOptions()
options.add_argument("--headless")  # Без UI (можно убрать)
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

# Инициализируем WebDriver (путь к Edge WebDriver должен быть установлен)
driver = webdriver.Edge(service=Service("msedgedriver.exe"), options=options)

# Открываем страницу с видео
url = "http://maps.ufanet.ru/sterlitamak#1659331721AYT463"
driver.get(url)
time.sleep(5)  # Даем странице загрузиться

# Получаем логи сетевых запросов
logs = driver.get_log("performance")

# Ищем ссылки на видео
video_links = []
for log in logs:
    msg = log["message"]
    if "m3u8" in msg or "mp4" in msg or "webm" in msg:
        print("🔹 Найден видео URL:", msg)
        video_links.append(msg)

# Закрываем браузер
driver.quit()

# Выводим найденные ссылки
if video_links:
    print("\n🎥 Потенциальные ссылки на потоковое видео:")
    for link in video_links:
        print(link)
else:
    print("❌ Видео-поток не найден.")
