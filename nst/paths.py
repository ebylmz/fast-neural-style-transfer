import os

# Base Directories
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")

# Data Subdirectories
STYLE_IMAGES_DIR = os.path.join(DATASET_DIR, "style_images")
CONTENT_IMAGES_DIR = os.path.join(DATASET_DIR, "content_images")
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "train") 

def get_style_image_path(style_id):
    return os.path.join(STYLE_IMAGES_DIR, f"{style_id}.jpg")

def get_model_save_path(style_id, content_weight, style_weight, tv_weight):
    filename = f"{style_id}_cw{content_weight}_sw{style_weight}_tw{tv_weight}.pth"
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, filename)

def get_log_dir(style_id, content_weight, style_weight, tv_weight):
    model_filename = f"{style_id}_cw{content_weight}_sw{style_weight}_tw{tv_weight}"
    return os.path.join(RUNS_DIR, model_filename)
