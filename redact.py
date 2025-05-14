import os

# Пути к файлам разметки
labels_dir = r"F:\project1\training_data\train"

# Переопределение классов: 0 - человек, 1 - оставить, 2 - оставить, 3 -> 2
class_mapping = {0: 0, 9: 2, 2: 1}

# Обрабатываем каждый .txt файл
for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        label_path = os.path.join(labels_dir, label_file)

        # Читаем содержимое файла
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                cls = int(parts[0])
                if cls in class_mapping:
                    parts[0] = str(class_mapping[cls])  # Заменяем класс, если он в словаре
                    new_lines.append(" ".join(parts) + "\n")

        # Перезаписываем файл с обновлёнными метками
        with open(label_path, "w") as f:
            f.writelines(new_lines)

print("Все .txt файлы исправлены.")
