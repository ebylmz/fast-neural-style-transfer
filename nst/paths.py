import os

# Adjust the ROOT_DIR as needed
# If running in Google Colab, you might want to set it to your Google Drive path
# Uncomment the next line if you are using Google Colab and have mounted your Google Drive
# ROOT_DIR = os.path.abspath('/content/drive/MyDrive/nst')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RUNS_DIR = os.path.join("/content/runs")

# Data Subdirectories
STYLE_IMAGES_DIR = os.path.join(DATASET_DIR, "style_images")
CONTENT_IMAGES_DIR = os.path.join(DATASET_DIR, "content_images")

def get_style_image_path(style_id):
    return os.path.join(STYLE_IMAGES_DIR, f"{style_id}.jpg")

def get_model_save_path(style_id, content_weight, style_weight, tv_weight):
    filename = f"{style_id}_cw{content_weight}_sw{style_weight}_tw{tv_weight}.pth"
    os.makedirs(MODEL_DIR, exist_ok=True)
    return os.path.join(MODEL_DIR, filename)

def get_log_dir(style_id, content_weight, style_weight, tv_weight):
    model_filename = f"{style_id}_cw{content_weight}_sw{style_weight}_tw{tv_weight}"
    return os.path.join(RUNS_DIR, model_filename)
