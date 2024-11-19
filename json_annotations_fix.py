# Открытие и чтение JSON-файла
import json

file_path = "C:\\main\\t2\\logo_detection\\marked\\tele2_dataset_2\\annotations\\instances_Validation.json"  # Укажите путь к JSON-файлу
with open(file_path, "r") as file:
    data = json.load(file)

data['categories'].append({
    "id": 0,
    "name": "background",
    "supercategory": ""
})

# Создаём список всех image_id
image_ids = {img['id'] for img in data['images']}

# Находим image_id, для которых отсутствуют аннотации
annotated_image_ids = {ann['image_id'] for ann in data['annotations']}
images_without_annotations = image_ids - annotated_image_ids

# Добавляем фиктивные аннотации для изображений без объектов
next_annotation_id = max(ann['id'] for ann in data['annotations']) + 1
for image_id in images_without_annotations:
    data['annotations'].append({
        "id": next_annotation_id,
        "image_id": image_id,
        "category_id": 0,  # Класс для "нулевых" объектов (если используется)
        "segmentation": [],
        "bbox": [0, 0, 0, 0],
        "area": 0,
        "iscrowd": 0,
    })
    next_annotation_id += 1

# Сохраняем обновленный JSON
output_file = "C:\\main\\t2\\logo_detection\\marked\\tele2_dataset_2\\annotations\\instances_Validation_fixed.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Фиктивные аннотации добавлены. Сохранено в {output_file}")
