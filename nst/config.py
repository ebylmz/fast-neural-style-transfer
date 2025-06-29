import os
from nst import paths

IMAGE_SIZE = 256
BATCH_SIZE = 4

class StyleConfig:
    def __init__(self, style_id, style_name, style_weight, content_weight, tv_weight):
        self.style_id = style_id
        self.style_name = style_name
        self.style_image_path = paths.get_style_image_path(style_id)
        self.model_path = paths.get_model_save_path(style_id, content_weight, style_weight, tv_weight)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
    
starry_night_cfg = StyleConfig(paths.MODEL_DIR, "starry_night", "Starry Night", style_weight=4e5, content_weight=2e0, tv_weight=2e0),
mosaic_cfg_cfg = StyleConfig(paths.MODEL_DIR, "mosaic", "Mosaic", style_weight=4e5, content_weight=2e0, tv_weight=2e0),
crystal_grove_cfg = StyleConfig(paths.MODEL_DIR, "crystal_grove", "Crystal Grove", style_weight=9e5, content_weight=2e0, tv_weight=2e0),
la_muse_cfg = StyleConfig(paths.MODEL_DIR, "la_muse", "La Muse", style_weight=9e5, content_weight=2e0, tv_weight=2e0),
candy_cfg = StyleConfig(paths.MODEL_DIR, "candy", "Candy", style_weight=9e5, content_weight=2e0, tv_weight=2e0),

def get_style_configs():
     return [starry_night_cfg, mosaic_cfg_cfg, crystal_grove_cfg, la_muse_cfg, candy_cfg]


def generate_training_config(
    style_config,
    lr=1e-3,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    num_epochs=1,
    enable_tensorboard=True,
    image_log_interval=100,
    console_log_interval=100,
):
    log_dir = paths.get_log_dir(
        style_config.style_id,
        style_config.content_weight,
        style_config.style_weight,
        style_config.tv_weight,
    )

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
            "style_image": style_config.style_image_path,
            "train_dataset": paths.DATASET_DIR,
            "model_save": style_config.model_path
        },
        "logging": {
            "log_dir": log_dir,
            "enable_tensorboard": enable_tensorboard,
            "image_log_interval": image_log_interval,
            "console_log_interval": console_log_interval
        }
    }
