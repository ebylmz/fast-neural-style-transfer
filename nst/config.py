import os

from nst.utils import generate_training_config

IMAGE_SIZE = 256
BATCH_SIZE = 4

DATASET_DIR = "data" # TODO: Fix the path
MODEL_DIR = "model"
TRAIN_IMGS_DIR = "" # TODO: Fix the path
STYLE_IMGS_DIR = os.join(DATASET_DIR, "style-images")
CONTENT_IMGS_DIR = os.join(DATASET_DIR, "content-images")

def generate_model_save_path(model_dir, style_id, content_weight, style_weight, tv_weight):
        filename = f"{style_id}_cw{content_weight}_sw{style_weight}_tw{tv_weight}.pth"
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)

class StyleConfig:
    def __init__(self, model_dir, style_id, style_name, style_weight, content_weight, tv_weight):
        self.style_id = style_id
        self.style_name = style_name
        self.style_image_path = os.join(STYLE_IMGS_DIR, f"{style_id}.jpg")
        self.model_path = generate_model_save_path(model_dir, style_id, content_weight, style_weight, tv_weight)
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_weight = tv_weight
    
starry_night = StyleConfig("starry_night", "Starry Night", style_weight=4e5, content_weight=2e0, tv_weight=2e0),
mosaic_cfg = StyleConfig("mosaic", "Mosaic", style_weight=4e5, content_weight=2e0, tv_weight=2e0),
crystal_grove = StyleConfig("crystal_grove", "Crystal Grove", style_weight=9e5, content_weight=2e0, tv_weight=2e0),
la_muse = StyleConfig("la_muse", "La Muse", style_weight=9e5, content_weight=2e0, tv_weight=2e0),
candy = StyleConfig("candy", "Candy", style_weight=9e5, content_weight=2e0, tv_weight=2e0),

def get_style_configs():
     return [starry_night, mosaic_cfg, crystal_grove, la_muse, candy]


def generate_training_config(
    style_config,
    train_dataset_path,
    model_save_dir,
    lr=1e-3,
    image_size=256,
    batch_size=4,
    num_epochs=1,
    enable_tensorboard=True,
    image_log_interval=100,
    console_log_interval=100,
    log_base_dir="runs",
):
    
    style_image_path = os.path.join(STYLE_IMGS_DIR, f"{style_config.style_id}.jpg")
    model_save_path = generate_model_save_path(model_save_dir, style_config)

    log_dir = f"{log_base_dir}/{os.path.splitext(os.path.basename(model_save_path))[0]}"

    return {
        "training": {
            "image_size": image_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": lr
        },
        "weights": {
            "style": style_config.style_weight,
            "content": style_config.content_weight,
            "tv": style_config.tv_weight
        },
        "paths": {
            "style_image": style_image_path,
            "train_dataset": train_dataset_path,
            "model_save": model_save_path
        },
        "logging": {
            "log_dir": log_dir,
            "enable_tensorboard": enable_tensorboard,
            "image_log_interval": image_log_interval,
            "console_log_interval": console_log_interval
        }
    }