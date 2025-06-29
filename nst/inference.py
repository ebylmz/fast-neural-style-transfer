from PIL import Image
import torch
import matplotlib.pyplot as plt
from nst import utils 

def resize_aspect_ratio(image, new_w):
    w, h = image.size
    new_h = int(h * new_w / w)
    return image.resize((new_w, new_h), Image.LANCZOS)

def stylize(
    model: torch.nn.Module,
    content_image: Image.Image,
    scaling_width: int = 1024
) -> Image.Image:
    """
    Stylizes a given PIL content image using a trained model.
    """
    model.eval()
    orig_size = content_image.size  # (W, H)

    # Optionally resize for faster stylization
    image_resized = resize_aspect_ratio(content_image, scaling_width)
    
    # Preprocessing
    transform = utils.get_transform(image_size=None, normalize=True, is_255_range=False)
    # input_tensor = transform(image_resized).unsqueeze(0).to(device)
    input_tensor = transform(image_resized).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0).cpu().numpy()
        output_np = utils.post_process_image(output_tensor)
        output_image = Image.fromarray(output_np)
    # Resize back to original
    return output_image.resize(orig_size, Image.LANCZOS)


def stylize_from_path(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    device: torch.device,
    image_width: int = 500,
    display: bool = True,
):
    """
    Loads an image from path, stylizes it, saves to disk, and optionally displays.
    """
    # Load and resize input image
    input_image = utils.load_style_image(
        image_path=input_path,
        device=device,
        image_size=image_width,
        batch_size=1,
        normalize=False  # we normalize manually in `stylize`
    )
    input_pil = utils.tensor_to_pil(input_image.squeeze(0))  # Convert back to PIL

    # Stylize
    output_pil = stylize(model, input_pil, device)

    # Save and optionally show
    output_pil.save(output_path)
    print(f"Stylized image saved to: {output_path}")

    if display:
        plt.imshow(output_pil)
        plt.axis("off")
        plt.title("Stylized Image")
        plt.show()
