# Кастомный Trainer для использования AlbumentationsMapper
from albumentations_mapper import AlbumentationsMapper
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=AlbumentationsMapper(cfg, is_train=True))

