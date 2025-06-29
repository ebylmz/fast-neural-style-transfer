import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# ImageNet normalization constants for images scaled to [0, 1] range
IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])  # Mean values for R, G, B channels
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

# ImageNet normalization constants for images in [0, 255] range
IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])

# Neutral standard deviation for 255-range normalization (no scaling, just mean subtraction)
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])


def get_transform(image_size=None, normalize=True, is_255_range=False):
    transform_list = []

    if image_size:
        transform_list.append(transforms.Resize(image_size))
        transform_list.append(transforms.CenterCrop(image_size))

    transform_list.append(transforms.ToTensor()) # Scales to float32 [0, 1]

    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x * 255))

    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
            if is_255_range else
            transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        )

    return transforms.Compose(transform_list)

# Custom Image Folder Dataset, since torchvision.datasets.ImageFolder loads labeled data
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=None):
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if max_images:
            self.image_paths = self.image_paths[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def load_style_image(image_path, device, image_size=None, batch_size=1, normalize=True, is_255_range=False):
    image = Image.open(image_path).convert('RGB')

    if isinstance(image_size, int):
        w, h = image.size
        new_w = image_size
        new_h = int(h * (new_w / w))
        image = image.resize((new_w, new_h), Image.LANCZOS)
    elif isinstance(image_size, tuple):
        image = image.resize(image_size, Image.LANCZOS)

    transform = get_transform(image_size=None, normalize=normalize, is_255_range=is_255_range)
    image_tensor = transform(image).to(device)  # Shape: [3, H, W]
    image_tensor = image_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # Shape: [B, 3, H, W]
    return image_tensor

def post_process_image(tensor_img):
    assert isinstance(tensor_img, np.ndarray), f'Expected numpy array, got {type(tensor_img)}'
    assert tensor_img.shape[0] == 3, f'Expected 3 channels, got shape {tensor_img.shape}'

    mean = IMAGENET_MEAN_1.reshape(3, 1, 1)
    std = IMAGENET_STD_1.reshape(3, 1, 1)

    # De-normalize
    img = tensor_img * std + mean

    # Clamp to valid range and convert to uint8
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    # Convert from CHW (PyTorch) to HWC (standard image format)
    img = np.transpose(img, (1, 2, 0))
    return img