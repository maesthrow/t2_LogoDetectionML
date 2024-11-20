from train import setup_cfg, train_model, model_evaluation, fine_tune_model
from train_new import train_model_new, evaluate_model_new, fine_tune_model_new
from train_yolo import train_yolo_model

if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()

    # train_model()
    # fine_tune_model()
    # model_evaluation()

    # train_model_new()
    # fine_tune_model_new()
    # evaluate_model_new("logo_val_new")

    train_yolo_model()

