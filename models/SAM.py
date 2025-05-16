import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.visualization import show_mask
from inference_pipeline import MultiModelPipeline
from transformers import SamModel


class SAM(MultiModelPipeline):

    def __init__(self, processor):
        super().__init__()  # Call the parent class constructor, if needed
        self.processor = processor
        self.model = SamModel.from_pretrained("facebook/sam-vit-base")
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

    def trainable_parameters(self):
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)"
        )
        return trainable_params, total_params

    def visualize(self, batch, pred_mask, normalize):
        image_tensor = batch["pixel_values"]
        model_input_boxes = batch["input_boxes"]
        model_pixel_values = batch["pixel_values"]

        if normalize:
            mean_norm = torch.tensor(
                self.processor.image_processor.image_mean, device=image_tensor.device
            ).view(3, 1, 1)
            std_norm = torch.tensor(
                self.processor.image_processor.image_std, device=image_tensor.device
            ).view(3, 1, 1)
            image_for_plot = image_tensor * std_norm + mean_norm
            image_for_plot = torch.clamp(image_for_plot, 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        img = model_pixel_values.squeeze(0)
        mean = torch.tensor(
            self.processor.image_processor.image_mean, device=img.device
        ).view(3, 1, 1)
        std = torch.tensor(
            self.processor.image_processor.image_std, device=img.device
        ).view(3, 1, 1)
        img_denorm = img * std + mean
        img_clipped = torch.clamp(img_denorm, 0, 1)
        img_np = img_clipped.permute(1, 2, 0).cpu().numpy()

        axes[0].imshow(img_np)
        axes[0].set_title("Immagine originale")
        axes[0].axis("off")
        if (
            model_input_boxes.shape[0] > 0 and model_input_boxes.shape[1] > 0
        ):  # Controlla se ci sono box
            # Assumiamo batch_size=1 e prendiamo il primo box
            box_coords = model_input_boxes[0, 0].cpu().numpy()
            x_min, y_min, x_max, y_max = box_coords
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[0].add_patch(rect)

        axes[1].imshow(img_np)

        if "show_mask" in globals():
            show_mask(pred_mask, axes[1], category=1)
        else:
            axes[1].imshow(pred_mask, alpha=0.5, cmap="Blues")
        axes[1].set_title("Immagine con Maschera Predetta")
        axes[1].axis("off")
        # Disegna l'input box sull'immagine con la maschera
        if (
            model_input_boxes.shape[0] > 0 and model_input_boxes.shape[1] > 0
        ):  # Controlla se ci sono box
            box_coords = model_input_boxes[0, 0].cpu().numpy()
            x_min, y_min, x_max, y_max = box_coords
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
            )
            axes[1].add_patch(rect)
        plt.tight_layout()
        plt.show()

    def predict_SAM(
        self,
        batch,
        image_embeddings=None,
        visualize_prediction=True,
        checkpoint_dir=None,
    ):
        predicted_mask = self.predict(
            self.model,
            model_name="SAM",
            batch=batch,
            image_embeddings=image_embeddings,
            threshold=0.5,
            processor=self.processor,
            binarize=True,
            visualize_prediction=visualize_prediction,
            checkpoint_dir=checkpoint_dir,
        )

        if visualize_prediction:
            self.visualize(batch, predicted_mask, normalize=True)
