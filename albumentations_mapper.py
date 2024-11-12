import albumentations
import torch
import albumentations as A

from detectron2.data import DatasetMapper, detection_utils


# Кастомный DatasetMapper для использования Albumentations
class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Blur(blur_limit=3, p=0.2)
        ])

    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")

        # Применение аугментаций
        augmented = self.transform(image=image)
        dataset_dict["image"] = torch.as_tensor(augmented["image"].copy().transpose(2, 0, 1))

        return dataset_dict
