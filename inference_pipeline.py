import torch
import os
from torch.nn.functional import sigmoid
from utils.metrics import binary_dice


class MultiModelPipeline:
    def __init__(self, device="cuda", eval=False):
        """
        Args:
            model_cls: callable to instantiate a new model (e.g., SamModel.from_pretrained)
            processor_cls: callable to load the processor (e.g., SamProcessor.from_pretrained)
            model_names (list): list of model identifiers (e.g., ["base", "ft1", "ft2"])
            par_dir (str): path to directory with checkpoints (e.g., 'checkpoints/base.pth')
            device (str): "cuda" or "cpu"
        """

        self.device = device
        self.models = {}
        self.processors = {}
        self.eval = eval
        self.dice = None
        self.prob_mask = None

    def register_model(self, name, model, processor):
        self.models[name] = model.to(self.device).eval()

    @torch.no_grad()
    def resize_mask(self, batch, outputs, processor, labels=False):
        original_image_sizes = [(1024, 1024)]  # lista di tuple
        reshaped_input_sizes = [(1024, 1024)]

        upscaled_masks_logits = processor.post_process_masks(
            outputs.pred_masks.cpu(),  # Passa i logit a bassa risoluzione (spostati su CPU se necessario)
            original_image_sizes,  # Dimensioni originali dell'immagine prima del processing di SAM
            reshaped_input_sizes,  # Dimensioni dell'immagine dopo il resize di SAM (prima del padding)
            binarize=False,  # Vogliamo i logit, non maschere binarizzate direttamente qui
        )
        if isinstance(upscaled_masks_logits, list):
            upscaled_masks_logits = torch.stack(
                upscaled_masks_logits
            )  # shape: (1, 1, H, W)
        upscaled_masks_logits = upscaled_masks_logits.squeeze(0)

        pred_mask_probs = torch.sigmoid(upscaled_masks_logits.squeeze(1))
        if labels:
            pred_mask_bin = (pred_mask_probs > 0.5).float()
            return pred_mask_bin  # Binarizza le probabilità
        else:
            return pred_mask_probs  # Restituisci le probabilità

    def predict(
        self,
        model,
        model_name=None,
        image_embeddings=None,
        batch=None,
        processor=None,
        threshold=0.5,
        binarize=True,
        visualize_prediction=True,
        checkpoint_dir=None,
    ):
        if model_name not in self.models:
            if checkpoint_dir is None:
                raise ValueError("checkpoint_dir must be provided to load model.")

        checkpoint_path = checkpoint_dir
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.eval()

        if model_name == "SAM":

            model_input_boxes = batch["input_boxes"]
            model_pixel_values = batch["pixel_values"]
            ground_truth_masks_from_batch = batch["ground_truth_mask"].float()
            outputs = model(
                pixel_values=model_pixel_values,
                input_boxes=model_input_boxes,
                image_embeddings=image_embeddings,
                multimask_output=False,
            )
            pred_mask_probs = self.resize_mask(batch, outputs, processor)
            self.prob_mask = pred_mask_probs
            if binarize:
                predicted_mask = (pred_mask_probs > threshold).squeeze(1)  # [B, H, W]

                if self.eval:
                    dice = binary_dice(
                        predicted_mask, ground_truth_masks_from_batch.squeeze(1)
                    )
                    self.dice = dice

            else:
                predicted_mask.squeeze(1)

            return predicted_mask
