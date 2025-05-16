import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # Aggiungi questa riga all'inizio del file
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torch.nn.functional as F
import random
from torchvision.transforms.functional import resize

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class COCOSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        year: str = "2017",
        processor=None,
        resize=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.year = year
        self.processor = processor
        self.resize = resize
        ann_file_name = f"instances_train2017.json"
        self.ann_file = os.path.join(root_dir, "annotations", ann_file_name)
        print(f"Annotation file path: {self.ann_file}")
        self.images_dir = os.path.join(root_dir, "images")

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        self.coco = COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def get_bounding_box(self, ground_truth_map):
        if len(ground_truth_map.shape) > 2:
            ground_truth_map = np.squeeze(ground_truth_map)

        if len(ground_truth_map.shape) != 2:
            raise ValueError(
                f"Expected 2D ground_truth_map, but got shape {ground_truth_map.shape}"
            )

        y_indices, x_indices = np.where(ground_truth_map == 1)

        if len(y_indices) != 0 and len(x_indices) != 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)

            H, W = ground_truth_map.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))

            return [x_min, y_min, x_max, y_max]
        else:

            return [0, 0, 1, 1]

    def preprocess_mask(self, mask):
        mask_np = mask.cpu().numpy()

        if mask_np.ndim > 2:
            mask_np = np.squeeze(mask_np)

        if mask_np.ndim != 2:
            raise ValueError(
                f"Mask should be 2D after squeezing, but got shape {mask_np.shape}"
            )

        # binary_mask_np = (mask_np == category).astype(np.uint8)

        pil_mask_image = Image.fromarray(mask_np * 255)
        pil_mask_resized = pil_mask_image.resize((1024, 1024), Image.NEAREST)
        mask_array = (np.array(pil_mask_resized) / 255.0).astype(np.uint8)
        return mask_array

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.images_dir, img_info["file_name"])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"ERROR: Image file not found at {image_path} for image ID {img_id}")
            raise

        ann_ids = self.coco.getAnnIds(imgIds=img_info["id"])
        anns_for_image = self.coco.loadAnns(ann_ids)

        ground_truth_mask = np.zeros(
            (img_info["height"], img_info["width"]), dtype=np.uint8
        )

        for ann in anns_for_image:
            if ann.get("iscrowd", 0) == 0:
                ground_truth_mask = np.maximum(
                    ground_truth_mask, self.coco.annToMask(ann) * ann["category_id"]
                )
        ground_truth_mask_tensor = (
            torch.as_tensor(ground_truth_mask, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        resized_ground_truth_mask = self.preprocess_mask(ground_truth_mask_tensor)
        # print(f"Resized ground truth mask shape: {resized_ground_truth_mask.shape}")
        # print(f"resized_ground_truth_mask: {resized_ground_truth_mask}")
        prompt = self.get_bounding_box(resized_ground_truth_mask)
        image = image.resize((1024, 1024), resample=Image.BILINEAR)

        if hasattr(self, "processor") and self.processor:
            inputs = self.processor(
                image,
                input_boxes=[[prompt]],
                masks=ground_truth_mask,
                return_tensors="pt",
            )
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        else:
            raise ValueError(
                "Processor is not defined. Please provide a valid processor."
            )
        inputs["ground_truth_mask"] = resized_ground_truth_mask

        return inputs


def dataloader(
    dataset,
    split: str = "train",
    year: str = "2017",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    # print(f"Attempting to load COCO dataset from: {root_dir} for {split}{year} split.")

    # if not os.path.exists(root_dir):
    #     raise FileNotFoundError(
    #         f"Dataset root directory not found at {root_dir}. Please check the path."
    #     )

    # image_transform = transforms.Compose(
    #     [
    #         transforms.Resize(img_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    # class PilToLongTensor(object):
    #     def __call__(self, pic):
    #         if not isinstance(pic, Image.Image):
    #             if isinstance(pic, torch.Tensor):
    #                 pic = transforms.ToPILImage()(pic)
    #             elif isinstance(pic, np.ndarray):
    #                 pic = Image.fromarray(pic)
    #             else:
    #                 raise TypeError(
    #                     f"Input 'pic' should be PIL Image, torch.Tensor or np.ndarray. Got {type(pic)}"
    #                 )
    #         return torch.as_tensor(np.array(pic, dtype=np.int64))

    # target_transform_pil = transforms.Compose(
    #     [
    #         transforms.Resize(
    #             img_size, interpolation=transforms.InterpolationMode.NEAREST
    #         ),
    #         PilToLongTensor(),
    #     ]
    # )

    # dataset = COCOSegmentationDataset(
    #     root_dir=root_dir,
    #     split=split,
    #     year=year,
    #     transform=image_transform,
    #     target_transform=target_transform_pil,
    # )
    # print(
    #     f"COCOSegmentationDataset created with {len(dataset)} samples for {split}{year} split."
    # )
    set_seed(42)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if split == "train" else False,
    )
    print(f"DataLoader created for {split}{year}. Number of batches: {len(dataloader)}")

    return dataloader


if __name__ == "__main__":
    COCO_ROOT_DIR = "data\\train2017\\"

    if COCO_ROOT_DIR == "data\\train2017" or not os.path.exists(COCO_ROOT_DIR):
        print("=" * 70)
        print("WARNING: Please update 'COCO_ROOT_DIR' in the example section with the")
        print("         actual path to your COCO dataset and ensure it exists.")
        print("         Skipping dataloader example.")
        print("=" * 70)
    else:
        print(f"Using COCO_ROOT_DIR: {COCO_ROOT_DIR}")
        try:
            dataset = COCOSegmentationDataset(
                root_dir=COCO_ROOT_DIR,
                split="train",
                year="2017",
                transform=None,
                target_transform=None,
            )
            train_dataloader = dataloader(
                dataset,
                split="train",
                year="2017",
                batch_size=2,
                shuffle=True,
            )

            print(f"\nSuccessfully created train_dataloader.")
            batch = next(iter(train_dataloader))
            for key, value in batch.items():
                print(f"{key}: {value}")
            print("\nExample dataloader iteration finished.")

        except FileNotFoundError as e:
            print(f"\nERROR: Could not load dataset. FileNotFoundError: {e}")
            print(
                "Please ensure the COCO dataset is correctly placed and paths are correct."
            )
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()
