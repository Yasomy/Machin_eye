import os
import zipfile
import shutil

def create_training_zip(images_dir, labels_dir, config_dir, output_zip="output2.zip"):
    temp_dir = "temp_zip_dir"
    obj_dir = os.path.join(temp_dir, "obj_Train_data")
    
    # Очистка или создание временной папки
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(obj_dir)

    # Копируем изображения
    for file in os.listdir(images_dir):
        if file.lower().endswith(".jpg"):
            shutil.copy(os.path.join(images_dir, file), obj_dir)

    # Копируем разметку
    for file in os.listdir(labels_dir):
        if file.lower().endswith(".txt"):
            shutil.copy(os.path.join(labels_dir, file), obj_dir)

    # Копируем конфиги (Train.txt, obj.names, obj.data)
    for file in ["Train.txt", "obj.names", "obj.data"]:
        src = os.path.join(config_dir, file)
        if os.path.isfile(src):
            shutil.copy(src, temp_dir)

    # Упаковываем всё в ZIP
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, temp_dir)
                zipf.write(abs_path, rel_path)

    # Очистка временной директории
    shutil.rmtree(temp_dir)
    print(f"✅ ZIP архив создан: {output_zip}")


create_training_zip(
    images_dir=r"F:\project1\training_data\images",
    labels_dir=r"F:\project1\training_data\labels",
    config_dir=r"F:\project1\training_data"
)
