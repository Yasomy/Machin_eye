import os

def save_image_paths(folder_path, output_file="image_names.txt", prefix="data/obj_Train_data/", extensions=(".jpg", ".jpeg", ".png")):
    output_path = os.path.join(folder_path, output_file)
    with open(output_path, "w") as f:
        for file in sorted(os.listdir(folder_path)):
            if file.lower().endswith(extensions):
                f.write(f"{prefix}{file}\n")

# Пример использования:
save_image_paths(r"F:\project1\training_data\images9")
