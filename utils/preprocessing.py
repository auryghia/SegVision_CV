from tensorflow import keras
import os
import numpy as np
import keras
from PIL import Image


class DataPreprocess(keras.utils.Sequence):
    def __init__(self, images_path, masks_path, batch_size, image_size=(128, 128)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.filenames = self._get_matching_filenames()

    def _get_matching_filenames(self):
        image_ids = {os.path.splitext(f)[0] for f in os.listdir(self.images_path)}
        mask_ids = {os.path.splitext(f)[0] for f in os.listdir(self.masks_path)}
        return sorted(list(image_ids & mask_ids))  # Sorted for consistent order

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.filenames[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        images, masks = [], []

        for file_id in batch_files:
            img = self._load_image(os.path.join(self.images_path, file_id + ".jpg"))
            mask = self._load_mask(os.path.join(self.masks_path, file_id + ".jpg"))

            if img.shape != (self.image_size[0], self.image_size[1], 3):
                continue  # Skip invalid images

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def _load_image(self, path):
        img = Image.open(path).resize(self.image_size)
        img_array = np.array(img).astype("float32") / 255.0
        return img_array

    def _load_mask(self, path):
        mask = Image.open(path).resize(self.image_size)
        mask_array = np.array(mask).astype("float32") / 255.0
        return mask_array
