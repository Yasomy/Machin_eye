import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # pip install deep_sort_realtime
import random


class ObjectDetectionStream:
    def __init__(self, stream_url, input_size=1920):
        """
        Инициализация детектора для видеопотока с трекингом.
        :param stream_url: URL видеопотока
        :param input_size: размер кадра для предсказания модели
        """
        self.stream_url = stream_url
        self.input_size = input_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔹 Используем устройство: {self.device.upper()}")
        self.model = self.load_model()
        # Если модель предоставляет словарь с названиями классов
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        """
        Загружает модель YOLO и переводит ее на нужное устройство.
        """
        model = YOLO("best_x4.pt").to(self.device)
        # При необходимости можно выполнить fuse() для ускорения:
        model.fuse()
        return model

    def predict(self, frame):
        """
        Выполняет предсказание на данном кадре.
        :param frame: кадр для обработки (размер должен быть input_size x input_size)
        :return: результаты детекции
        """
        results = self.model(frame)
        return results

    def get_results(self, results, original_w, original_h):
        """
        Извлекает детекции для классов person (0) и transport (2)
        и преобразует координаты обратно к исходному разрешению.
        
        :param results: результаты модели
        :param original_w: исходная ширина кадра
        :param original_h: исходная высота кадра
        :return: (detections, person_count, transport_count)
          detections – numpy-массив, где каждая строка имеет вид:
                       [x1, y1, x2, y2, confidence, class_id]
        """
        detections = []
        person_count = 0
        transport_count = 0
        # Коэффициенты масштабирования для возврата координат к исходному разрешению
        scale_x = original_w / self.input_size
        scale_y = original_h / self.input_size

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls in [0, 2]:
                bbox = box.xyxy.cpu().numpy()  # формат [[x1, y1, x2, y2]]
                conf = box.conf.cpu().numpy()[0]
                x1 = bbox[0][0] * scale_x
                y1 = bbox[0][1] * scale_y
                x2 = bbox[0][2] * scale_x
                y2 = bbox[0][3] * scale_y
                detections.append([x1, y1, x2, y2, conf, cls])
                if cls == 0:
                    person_count += 1
                elif cls == 2:
                    transport_count += 1
        detections = np.array(detections) if detections else np.empty((0, 6))
        return detections, person_count, transport_count

    def draw_tracking_boxes(self, frame, tracked_objects):
        """
        Отрисовывает трекинговые боксы для объектов класса person с идентификатором (ID).
        :param frame: исходный кадр
        :param tracked_objects: список боксов с трекинговой информацией [x1, y1, x2, y2, track_id]
        :return: кадр с отрисованными бокcами
        """
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv2.putText(frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return frame

    def draw_transport_boxes(self, frame, detections):
        """
        Отрисовывает боксы для объектов класса transport.
        :param frame: исходный кадр
        :param detections: массив детекций [x1, y1, x2, y2, conf, class_id]
        :return: кадр с отрисованными бокcами для транспорта
        """
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 2:
                # Тонкая обводка: толщина 1, текст не отрисовываем
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        return frame

    def __call__(self):
        """
        Основной цикл обработки видеопотока.
        Выполняет предсказание, трекинг для объектов класса person через DeepSORT,
        отрисовку трекинговых боксов и боксов транспорта, вывод FPS и позволяет сохранять скриншоты.
        """
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print("❌ Ошибка: OpenCV не может открыть поток!")
            return

        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if original_w == 0 or original_h == 0:
            print("❌ Ошибка: неверный размер видео!")
            return

        num = 1  # номер для сохранения скриншотов
        prev_time = time.time()

        # Инициализируем DeepSORT трекер для объектов класса person
        # Обратите внимание: параметры можно подбирать, здесь приведён пример с n_init=5 и max_age=1000
        tracker = DeepSort(max_age=1000, n_init=15, max_cosine_distance=0.3, embedder="mobilenet", embedder_gpu=True)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("❌ Ошибка: не удалось получить кадр.")
                break

            # Изменяем размер для подачи в модель
            resized_frame = cv2.resize(frame, (self.input_size, self.input_size))
            results = self.predict(resized_frame)
            # Получаем детекции для классов person и transport
            detections, person_count, transport_count = self.get_results(results, original_w, original_h)

            # Подготавливаем список детекций для трекинга (только объекты класса person)
            # Для DeepSORT требуется формат [xmin, ymin, width, height, confidence]
            person_detections_deep = []
            for det in detections:
                if int(det[5]) == 0:
                    x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]
                    width = x2 - x1
                    height = y2 - y1
                    person_detections_deep.append([[x1, y1, width, height], conf])

            # Обновляем трекинг через DeepSORT. Если нет детекций, передаём пустой список.
            tracks = tracker.update_tracks(person_detections_deep, frame=frame)

            # Извлекаем боксы и идентификаторы треков
            tracked_objects = []
            for track in tracks:
                # Проверяем, подтверждён ли трек
                if not track.is_confirmed():
                    continue
                bbox = track.to_ltrb()  # возвращает [x1, y1, x2, y2]
                track_id = track.track_id
                tracked_objects.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])

            # Отрисовываем трекинговые боксы для persons
            frame = self.draw_tracking_boxes(frame, tracked_objects)
            # Отрисовываем боксы для транспорта
            frame = self.draw_transport_boxes(frame, detections)

            # Расчет FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(frame, f"People: {person_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            cv2.putText(frame, f"Transport: {transport_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

            cv2.imshow("YOLO Stream with DeepSORT (GPU)", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite(f'images/img{num}.png', frame)
                print("Изображение сохранено!")
                num += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    stream_url = (
        "http://46.191.199.12/1660720512DSH176/index.fmp4.m3u8?token=7acaece6c9fc4550a83d2ee1e4316e4e"
    )
    detector = ObjectDetectionStream(stream_url)
    detector()
