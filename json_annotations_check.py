import json

# Открытие и чтение JSON-файла
file_path = "C:\\main\\t2\\logo_detection\\marked\\tele2_dataset_2\\annotations\\instances_Train.json"  # Укажите путь к JSON-файлу
with open(file_path, "r") as file:
    data = json.load(file)

print(len(data.get("annotations", [])))

# Проверка объектов в секции annotations
filtered_annotations = [
    annotation for annotation in data.get("annotations", [])
    # if "segmentation" in annotation
]

# Вывод результатов
print("Annotations without 'segmentation':")
for annotation in filtered_annotations:
    print(annotation)
