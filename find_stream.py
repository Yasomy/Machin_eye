from selenium import webdriver
from selenium.webdriver.edge.service import Service
import json
import time

# Путь к WebDriver (укажите правильный)
service = Service("msedgedriver.exe")
options = webdriver.EdgeOptions()
options.add_argument("--headless")  # Можно убрать, если нужен UI
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Запускаем Edge
driver = webdriver.Edge(service=service, options=options)

# Включаем мониторинг сетевых запросов
driver.execute_cdp_cmd("Network.enable", {})

# Открываем страницу с камерой
url = "http://maps.ufanet.ru/sterlitamak#1659331721AYT463"
driver.get(url)
time.sleep(10)  # Ждём загрузку видео

# Получаем сетевые запросы
video_links = []

# Перехватываем запросы
def process_request(request):
    try:
        url = request["params"]["request"]["url"]
        if any(ext in url for ext in [".m3u8", ".mp4", ".webm", ".flv", ".ts"]):
            print("🔹 Найден видео URL:", url)
            video_links.append(url)
    except Exception as e:
        pass

# Подключаем обработку сетевых запросов
driver.execute_cdp_cmd("Network.setRequestInterception", {"patterns": [{"urlPattern": "*"}]})
driver.execute_script("""
    (function() {
        window.networkRequests = [];
        window.performance.getEntriesByType('resource').forEach(function(request) {
            if (request.name.includes('.m3u8') || request.name.includes('.mp4')) {
                window.networkRequests.push(request.name);
            }
        });
    })();
""")

time.sleep(5)  # Даем время для загрузки запросов

# Читаем перехваченные запросы
logs = driver.execute_script("return window.networkRequests;")

# Проверяем найденные ссылки
for log in logs:
    print("🔹 Найден видео URL:", log)
    video_links.append(log)

# Закрываем браузер
driver.quit()

# Выводим результат
if video_links:
    print("\n🎥 Потенциальные ссылки на потоковое видео:")
    for link in video_links:
        print(link)
else:
    print("❌ Видео-поток не найден.")
