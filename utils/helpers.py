from zipfile import ZipFile, BadZipFile
import os
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, target_class_index):
        """
        Args:
            embeddings: list of dicts with keys 'embedding', 'input_boxes', and 'ground_truth_mask'
            target_class_index: the class index you want to extract (after remapping)
        """
        self.embeddings = embeddings
        self.target_class_index = target_class_index

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        data = self.embeddings[idx]
        embedding_tensor = data["embedding"].last_hidden_state
        input_boxes = data["input_boxes"].float()
        gt_mask = data["ground_truth_mask"].long()

        # Create binary mask: 1 where equal to selected class, 0 elsewhere
        binary_mask = gt_mask == self.target_class_index

        return {
            "embedding": embedding_tensor,
            "input_boxes": input_boxes,
            "ground_truth_mask": binary_mask,  # shape: (H, W) or (1, H, W)
        }
