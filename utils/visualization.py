import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_mask(mask_np, ax, category=1, random_color=False):
    """
    Mostra la maschera per una specifica categoria, con overlay sulla figura.
    Args:
        mask_tensor (torch.Tensor): Maschera di input (potrebbe essere 4D).
        ax (matplotlib.axes.Axes): Asse su cui visualizzare la maschera.
        category (int): Categoria da visualizzare.
        random_color (bool): Se True, usa colore casuale.
    """
    mask_np = mask_np.cpu().numpy() if isinstance(mask_np, torch.Tensor) else mask_np
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)

    if mask_np.ndim != 2:
        raise ValueError(
            f"Mask should be 2D after squeezing, but got shape {mask_np.shape}"
        )

    binary_mask_np = mask_np == category
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    pil_mask_image = Image.fromarray(binary_mask_np * 255)

    # pil_mask_resized = pil_mask_image.resize((1024, 1024), Image.NEAREST)
    binary_mask_resized_float = np.array(pil_mask_image) / 255.0
    mask_image_colored = binary_mask_resized_float[..., None] * color.reshape(1, 1, -1)
    ax.imshow(mask_image_colored)


def show_image_with_mask(dataset, idx, normalize, processor):
    sample = dataset[idx]
    image_tensor = sample["pixel_values"]  # Tensor (C, H, W), es. (3, 1024, 1024)

    if normalize:
        mean_norm = torch.tensor(
            processor.image_processor.image_mean, device=image_tensor.device
        ).view(3, 1, 1)
        std_norm = torch.tensor(
            processor.image_processor.image_std, device=image_tensor.device
        ).view(3, 1, 1)
        image_for_plot = image_tensor * std_norm + mean_norm
        image_for_plot = torch.clamp(image_for_plot, 0, 1)

    image_for_plot = (
        image_for_plot.permute(1, 2, 0).cpu().numpy()
        if normalize == True
        else image_tensor.permute(1, 2, 0).cpu().numpy()
    )
    mask = sample["ground_truth_mask"]
    boxes_tensor = sample["input_boxes"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_for_plot)
    axes[0].set_title("Immagine con Ground Truth Box")
    axes[0].axis("off")

    if boxes_tensor.shape[0] > 0:
        box = boxes_tensor[0].cpu().numpy()
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        axes[0].add_patch(rect)

    axes[1].imshow(image_for_plot)
    show_mask(mask, axes[1], category=1)
    axes[1].set_title("Immagine con Maschera e GT Box (Cat. 1)")
    axes[1].axis("off")

    if boxes_tensor.shape[0] > 0:
        box = boxes_tensor[0].cpu().numpy()
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none"
        )
        axes[1].add_patch(rect)

    plt.tight_layout()
    plt.show()
