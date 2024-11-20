import cv2
from dotenv import load_dotenv

import detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

import os

from detectron2.utils.visualizer import Visualizer, ColorMode

load_dotenv()
# Путь к набору данных для обучения
dataset_path = os.getenv("DATASET_PATH_BASE")
train_json = os.path.join(dataset_path, "train\\result.json")  # Файл аннотаций
train_images = os.path.join(dataset_path, "train")  # Папка с изображениями
val_json = os.path.join(dataset_path, "val\\result.json")
val_images = os.path.join(dataset_path, "val")

# Регистрация датасета в формате COCO
register_coco_instances("logo_train", {}, train_json, train_images)
register_coco_instances("logo_val", {}, val_json, val_images)


# def setup_cfg(model_name):
#     cfg: CfgNode = get_cfg()
#     cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#     cfg.DATASETS.TRAIN = ("logo_train",)
#     cfg.DATASETS.TEST = ("logo_val",)
#     cfg.DATALOADER.NUM_WORKERS = 12
#     cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#     cfg.SOLVER.IMS_PER_BATCH = 4
#     cfg.SOLVER.BASE_LR = 0.00020  # Скорость обучения
#     cfg.SOLVER.MAX_ITER = 2000  # Количество итераций
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Количество классов: "старый логотип" и "новый логотип"
#     cfg.OUTPUT_DIR = f"./output/{model_name}"
#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#
#     return cfg


def setup_cfg(model_name, base_lr=0.00025, max_iter=1000, num_classes=1, pretrained_weights=None):
    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("logo_train",)
    cfg.DATASETS.TEST = ("logo_val",)
    cfg.DATALOADER.NUM_WORKERS = 12
    cfg.MODEL.WEIGHTS = pretrained_weights if pretrained_weights else detectron2.model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = base_lr  # Установлен более низкий learning rate
    cfg.SOLVER.MAX_ITER = max_iter  # Меньшее количество итераций для дотренировки

    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.8))
    cfg.SOLVER.GAMMA = 0.1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Количество классов: "старый логотип" и "новый логотип"

    cfg.MODEL.DEVICE = "cuda"

    cfg.OUTPUT_DIR = f"./output/{model_name}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def train_model():
    cfg = setup_cfg('model_t', base_lr=0.00025, max_iter=2000, num_classes=1)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def fine_tune_model():
    cfg = setup_cfg('model_t.3', base_lr=0.00005, max_iter=1000, pretrained_weights="./output/model_t.2/model_final.pth", num_classes=1)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)  # Установлено resume=True, чтобы продолжить обучение
    trainer.train()


def model_evaluation():
    cfg = setup_cfg('model_5.2', num_classes=2)
    # cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TEST = ("logo_val",)

    # Создание предиктора для использования обученной модели
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Порог уверенности предсказания
    predictor = DefaultPredictor(cfg)
    print(cfg.MODEL.WEIGHTS)
    # Загрузка тестового изображения и предсказание
    # test_image_path = "D:\\dev\\aston\\t2\\logo_detection\\new_logo_test\\30eafcf5-041a-42ac-8b81-fa185b21bb42.jpg"
    # test_image_path = "D:\\dev\\aston\\t2\\logo_detection\\new_logo_test\\T2_Digital_game-scaled.jpg"
    # test_image_path = "D:\\dev\\aston\\t2\\logo_detection\\new_logo_test\\e6d10894-28b0-4b34-8664-a80b7f943017.jpg"
    # test_image_path = "D:\\dev\\aston\\t2\\logo_detection\\new_logo_test\\f3ec4e75-ff0316cd1aeaaedde987726f67982f50.jpg"
    test_image_path = "D:\\dev\\aston\\t2\\logo_detection\\MLTest\\2C5AD058-099E-4675-AB90-F8B9C5918162.jpg"
    # test_image_path = "C:\\Users\\dmitr\\Downloads\\3dc135ae-cf8e-4a25-b253-21ed6bd74b82.jfif"
    img = cv2.imread(test_image_path)
    outputs = predictor(img)

    # Визуализация результатов с отображением названий классов
    v = Visualizer(img[:, :, ::-1], metadata={"thing_classes": ['"TELE2"', '"t2"']}, scale=1.2,
                   instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow("Detection", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

