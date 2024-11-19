import cv2

import detectron2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

import os

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer, ColorMode

# Путь к набору данных для обучения
dataset_path = "C:\\main\\t2\\logo_detection\\marked\\tele2_dataset_3"
train_json = os.path.join(dataset_path, "annotations\\instances_Train.json")  # Файл аннотаций
train_images = os.path.join(dataset_path, "images\\Train")  # Папка с изображениями
val_json = os.path.join(dataset_path, "annotations\\instances_Validation.json")
val_images = os.path.join(dataset_path, "images\\Validation")

# Регистрация датасета в формате COCO
register_coco_instances("logo_train_new", {}, train_json, train_images)
register_coco_instances("logo_val_new", {}, val_json, val_images)


def setup_cfg(model_name, base_lr=0.00025, max_iter=1000, num_classes=1, pretrained_weights=None):
    cfg: CfgNode = get_cfg()
    cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("logo_train_new",)
    cfg.DATASETS.TEST = ("logo_val_new",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = pretrained_weights if pretrained_weights else detectron2.model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = base_lr  # Установлен более низкий learning rate
    cfg.SOLVER.MAX_ITER = max_iter  # Меньшее количество итераций для дотренировки

    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.STEPS = (int(max_iter * 0.5), int(max_iter * 0.75))
    cfg.SOLVER.GAMMA = 0.1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.SOLVER.AMP.ENABLED = True

    cfg.MODEL.DEVICE = "cuda"

    cfg.MODEL.MASK_ON = True  # Включаем поддержку масок (полигоны)

    cfg.OUTPUT_DIR = f"./output_new/{model_name}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def setup_cfg_mask(model_name, base_lr=0.00025, max_iter=1000, num_classes=1, pretrained_weights=None):
    cfg: CfgNode = get_cfg()

    # Заменяем ResNet-50 на ResNet-101
    cfg.merge_from_file(detectron2.model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("logo_train_new",)
    cfg.DATASETS.TEST = ("logo_val_new",)
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = pretrained_weights if pretrained_weights else detectron2.model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

    # Уменьшение размера батча для экономии памяти
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter

    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.STEPS = (int(max_iter * 0.5), int(max_iter * 0.75))
    cfg.SOLVER.GAMMA = 0.1

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Включаем AMP для ускорения и экономии памяти
    cfg.SOLVER.AMP.ENABLED = True

    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.MASK_ON = True  # Включаем поддержку масок (полигоны)

    # Указываем папку для сохранения модели
    cfg.OUTPUT_DIR = f"./output_new/{model_name}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def train_model_new():
    cfg = setup_cfg_mask('model_resnet101', base_lr=0.0001, max_iter=4000, num_classes=1)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Оценка модели после обучения
    metrics = _evaluate_model(cfg, trainer.model, "logo_val_new")
    print(metrics)


def fine_tune_model_new():
    cfg = setup_cfg_mask(
        'model_resnet101.2',
        base_lr=0.00005,
        max_iter=1500,
        pretrained_weights="./output_new/model_resnet101/model_final.pth",
        num_classes=1
    )
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)  # Установлено resume=True, чтобы продолжить обучение
    trainer.train()

    # Оценка модели после обучения
    metrics = _evaluate_model(cfg, trainer.model, "logo_val_new")
    print(metrics)


def _evaluate_model(cfg, model, dataset_name):
    evaluator = COCOEvaluator(dataset_name, (), False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(model, val_loader, evaluator)


def evaluate_model_new(dataset_name):
    # Загрузка конфигурации
    cfg = setup_cfg('model_a', base_lr=0.0001, max_iter=4000, num_classes=1)

    # Указание датасета для валидации
    cfg.DATASETS.TEST = (dataset_name,)

    # Загрузка обученной модели
    model = DefaultTrainer.build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    final_weights_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    checkpointer.load(final_weights_path)  # Загрузка весов модели
    print(f"Загружена модель из {final_weights_path}")

    # Запуск оценки
    evaluator = COCOEvaluator(dataset_name, (), False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    metrics = inference_on_dataset(model, val_loader, evaluator)

    print("Результаты оценки:", metrics)
    return metrics


def model_evaluation_new():
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

