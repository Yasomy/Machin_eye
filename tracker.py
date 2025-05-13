import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
import sort  # (pip install filterpy lap)

# Глобальные переменные для работы с ROI
drawing = False
current_roi = None
rois = []
single_roi_mode = True
show_rois = True
ix, iy = -1, -1

def draw_menu(frame):
    H, W = frame.shape[:2]
    roi_mode_button = (W - 230, 10, W - 10, 40)
    roi_show_button = (W - 230, 50, W - 10, 80)

    cv2.rectangle(frame, (roi_mode_button[0], roi_mode_button[1]),
                  (roi_mode_button[2], roi_mode_button[3]), (50, 50, 50), -1)
    cv2.rectangle(frame, (roi_show_button[0], roi_show_button[1]),
                  (roi_show_button[2], roi_show_button[3]), (50, 50, 50), -1)

    mode_text = "ROI Mode: Single" if single_roi_mode else "ROI Mode: Multiple"
    show_text = "ROI: On" if show_rois else "ROI: Off"

    cv2.putText(frame, mode_text, (roi_mode_button[0] + 5, roi_mode_button[3] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, show_text, (roi_show_button[0] + 5, roi_show_button[3] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return roi_mode_button, roi_show_button

def select_roi(event, x, y, flags, param):
    global ix, iy, drawing, current_roi, rois, single_roi_mode, show_rois
    frame = param
    H, W = frame.shape[:2]
    roi_mode_button = (W - 230, 10, W - 10, 40)
    roi_show_button = (W - 230, 50, W - 10, 80)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Переключение режимов ROI
        if roi_mode_button[0] <= x <= roi_mode_button[2] and roi_mode_button[1] <= y <= roi_mode_button[3]:
            single_roi_mode = not single_roi_mode
            if single_roi_mode and rois:
                rois = [rois[-1]]
            print("ROI Mode:", "Single" if single_roi_mode else "Multiple")
            return
        if roi_show_button[0] <= x <= roi_show_button[2] and roi_show_button[1] <= y <= roi_show_button[3]:
            show_rois = not show_rois
            if not show_rois:
                rois.clear()
                print("ROI отключены, трекинг всего кадра")
            else:
                print("ROI включены")
            return
        # Начало рисования ROI
        drawing = True
        ix, iy = x, y
        current_roi = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_roi = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        temp = frame.copy()
        draw_menu(temp)
        cv2.rectangle(temp, (current_roi[0], current_roi[1]),
                      (current_roi[2], current_roi[3]), (0, 255, 0), 2)
        cv2.imshow("YOLO Stream with Tracking (GPU)", temp)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        current_roi = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        if single_roi_mode:
            rois = [current_roi]
        else:
            rois.append(current_roi)
        cv2.rectangle(frame, (current_roi[0], current_roi[1]),
                      (current_roi[2], current_roi[3]), (0, 255, 0), 2)
        draw_menu(frame)
        cv2.imshow("YOLO Stream with Tracking (GPU)", frame)

class ObjectDetectionStream:
    def __init__(self, stream_url, input_size=1920):
        self.stream_url = stream_url
        self.input_size = input_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔹 Используем устройство: {self.device.upper()}")
        self.model = self.load_model()

    def load_model(self):
        m = YOLO("best_x5.pt").to(self.device)
        m.fuse()
        return m

    def predict(self, frame):
        return self.model(frame)

    def get_results(self, results, w, h):
        dets = []
        pc = tc = 0
        sx, sy = w / self.input_size, h / self.input_size

        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls in (0, 1):
                coords = box.xyxy.cpu().numpy()[0]
                conf = float(box.conf.cpu().numpy()[0])
                x1, y1, x2, y2 = coords * np.array([sx, sy, sx, sy])
                dets.append([x1, y1, x2, y2, conf, cls])
                if cls == 0: pc += 1
                else:        tc += 1

        return np.array(dets), pc, tc

    def draw_tracking_boxes(self, frame, tracks):
        for x1, y1, x2, y2, tid in tracks:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv2.putText(frame, f"ID:{int(tid)}", (int(x1), int(y1)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        return frame

    def draw_transport_boxes(self, frame, dets):
        for x1, y1, x2, y2, _, cls in dets:
            if int(cls) == 1:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print("❌ Не удалось открыть поток")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == 0 or h == 0:
            print("❌ Неверный размер видео")
            return

        tracker = sort.Sort(max_age=10000, min_hits=5, iou_threshold=0.15)
        window = "YOLO Stream with Tracking (GPU)"
        cv2.namedWindow(window)
        global rois

        prev = time.time()
        roi_person_ids = set()
        roi_entries = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Не удалось получить кадр")
                break

            cv2.setMouseCallback(window, select_roi, frame.copy())
            img = cv2.resize(frame, (self.input_size, self.input_size))
            results = self.predict(img)
            dets, pc, tc = self.get_results(results, w, h)

            # Фильтрация по ROI (сохраняем все 6 полей!)
            if not show_rois:
                rois.clear()
            elif rois:
                filt = []
                pc2 = tc2 = 0
                for x1, y1, x2, y2, conf, cls in dets:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rois):
                        filt.append([x1, y1, x2, y2, conf, cls])
                        if cls == 0: pc2 += 1
                        else:        tc2 += 1
                dets = np.array(filt)
                pc, tc = pc2, tc2

            # Сброс счётчика, если сейчас нет ни одного ROI
            if not show_rois or not rois:
                roi_person_ids.clear()
                roi_entries = 0

            # Подготовка к трекингу людей
            people = dets[dets[:,5] == 0] if dets.size else np.empty((0,6))
            tracks = tracker.update(people[:,:5] if people.size else people)

            # Обновляем счётчик уникальных ID (только при наличии ROI)
            if show_rois and rois:
                for x1, y1, x2, y2, tid in tracks:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rois):
                        t = int(tid)
                        if t not in roi_person_ids:
                            roi_person_ids.add(t)
                            roi_entries += 1

            out = frame.copy()
            out = self.draw_tracking_boxes(out, tracks)
            out = self.draw_transport_boxes(out, dets)

            now = time.time()
            fps = 1 / (now - prev) if now != prev else 0
            prev = now

            # Отрисовка текста
            cv2.putText(out, f"FPS: {int(fps)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(out, f"People: {pc}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(out, f"Transport: {tc}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            cv2.putText(out, f"Unique in ROI: {roi_entries}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # Рисуем ROI
            if show_rois and rois:
                for r in rois:
                    cv2.rectangle(out, (r[0],r[1]), (r[2],r[3]), (0,255,0), 2)

            draw_menu(out)
            cv2.imshow(window, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'images/img{frame_idx}.png', out)
                frame_idx += 1
                print("Изображение сохранено")
            elif key == ord('r'):
                rois.clear()
                roi_person_ids.clear()
                roi_entries = 0
                print("ROI сброшены, счётчик обнулён")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    stream_url = "http://46.191.199.12/1660720512DSH176/tracks-v1/index.fmp4.m3u8?token=4fd72a282a254700869cfa25870f9371"
    ObjectDetectionStream(stream_url)()
