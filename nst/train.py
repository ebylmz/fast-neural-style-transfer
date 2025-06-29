import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from nst.transform_net import TransformNet
import nst.perceptual_loss_net as perceptual
import nst.utils as utils


def train(dataloader, training_config):
    # Config sections
    training_params = training_config["training"]
    weight_params = training_config["weights"]
    paths = training_config["paths"]
    logging_cfg = training_config["logging"]

    # Setup device and initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_net = TransformNet().to(device)
    perceptual_net = perceptual.PerceptualLossNet().to(device)
    perceptual_net.eval()

    # Optimizer and loss function
    optimizer = optim.Adam(transform_net.parameters(), lr=training_params["lr"])
    mse_loss = nn.MSELoss(reduction="mean")

    # Load and process style image
    style_batch = utils.load_style_image(
        image_path=paths["style_image_path"],
        device=device,
        image_size=None,
        batch_size=training_params["batch_size"]
    )

    # Extract target style representations (Gram matrices)
    target_style_grams = [perceptual.gram_matrix(fmap) for fmap in perceptual_net(style_batch)]

    # TensorBoard Logging
    writer = None
    if logging_cfg["enable_tensorboard"]:
        os.makedirs(logging_cfg["log_dir"], exist_ok=True)
        writer = SummaryWriter(log_dir=logging_cfg["log_dir"])
        writer.add_hparams({
            "lr": training_params["lr"],
            "content_weight": weight_params["content_weight"],
            "style_weight": weight_params["style_weight"],
            "batch_size": training_params["batch_size"]
        }, {})

    # Training loop
    transform_net.train()
    for epoch in range(training_params["num_epochs"]):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{training_params['num_epochs']}")
        for batch_idx, content_batch in enumerate(progress_bar):
            optimizer.zero_grad()
            content_batch = content_batch.to(device)

            # Step 1: Generate stylized images
            stylized_batch = transform_net(content_batch)

            # Step 2: Extract features using perceptual network (VGG-16)
            content_features = perceptual_net(content_batch)
            stylized_features = perceptual_net(stylized_batch)

            # Step 3: Compute content loss (using relu2_2 layer)
            content_loss = weight_params["content_weight"] * mse_loss(content_features.relu2_2, stylized_features.relu2_2)

            # Step 4: Compute style loss using Gram matrices
            style_loss = 0.0
            stylized_grams = [perceptual.gram_matrix(fmap) for fmap in stylized_features]
            for stylized_gram, target_gram in zip(stylized_grams, target_style_grams):
                style_loss += mse_loss(stylized_gram, target_gram)
            style_loss = weight_params["style_weight"] * (style_loss / len(target_style_grams))

            # Step 5: Compute total variation loss - enforces image smoothness
            tv_loss = weight_params["tv_weight"] * perceptual.total_variation_loss(stylized_batch)

            # Step 6: Combine losses and backpropagate
            total_loss = content_loss + style_loss + tv_loss
            epoch_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # TensorBoard Logging
            if writer:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/Content", content_loss.item(), global_step)
                writer.add_scalar("Loss/Style", style_loss.item(), global_step)
                writer.add_scalar("Loss/TV", tv_loss.item(), global_step)

                if batch_idx % logging_cfg["image_log_interval"] == 0:
                    tensor_img = stylized_batch[0].detach().cpu().numpy()
                    img = utils.post_process_image(tensor_img)
                    img = np.transpose(img, (2, 0, 1))  # CHW for TensorBoard
                    writer.add_image("Stylized", img, global_step)

            # Console logging
            if batch_idx % logging_cfg["console_log_interval"] == 0:
                progress_bar.set_postfix({
                    "Total Loss": f"{total_loss.item():.2f}",
                    "Content Loss": f"{content_loss.item():.2f}",
                    "Style Loss": f"{style_loss.item():.2f}",
                    "Total Variance Loss": f"{tv_loss.item():.2f}"
                })

        # Epoch logging
        avg_epoch_loss = epoch_loss / len(dataloader)
        if writer:
            writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch)

    # Save the final model
    torch.save(transform_net.state_dict(), paths["model_save_path"])
    print(f"\nTraining complete. Model saved to {paths['model_save_path']}")

    return transform_net, optimizer
