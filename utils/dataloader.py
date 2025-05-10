import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO


class COCOSegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str = 'train', year: str = '2017', transform=None, target_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.year = year
        self.transform = transform
        self.target_transform = target_transform
        ann_file_name = f'instances_{split}{year}.json'
        self.ann_file = os.path.join(root_dir, 'annotations', ann_file_name)
        self.images_dir = os.path.join(root_dir, 'images', f'{split}{year}')

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        self.coco = COCO(self.ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]

        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.images_dir, img_info['file_name'])
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERROR: Image file not found at {image_path} for image ID {img_id}")
            raise

        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        anns_for_image = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns_for_image:
            if ann.get('iscrowd', 0) == 0:
                mask = np.maximum(mask, self.coco.annToMask(ann) * ann['category_id'])

        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = transforms.ToTensor()(mask).squeeze(0).long()

        return image, mask


def get_coco_dataloader(root_dir: str, split: str = 'train', year: str = '2017',
                        batch_size: int = 32, shuffle: bool = True,
                        num_workers: int = 4, pin_memory: bool = True,
                        img_size: tuple = (256, 256)):
    print(f"Attempting to load COCO dataset from: {root_dir} for {split}{year} split.")

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root directory not found at {root_dir}. Please check the path.")

    image_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class PilToLongTensor(object):
        def __call__(self, pic):
            if not isinstance(pic, Image.Image):
                if isinstance(pic, torch.Tensor):
                    pic = transforms.ToPILImage()(pic)
                elif isinstance(pic, np.ndarray):
                    pic = Image.fromarray(pic)
                else:
                    raise TypeError(f"Input 'pic' should be PIL Image, torch.Tensor or np.ndarray. Got {type(pic)}")
            return torch.as_tensor(np.array(pic, dtype=np.int64))

    target_transform_pil = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        PilToLongTensor()
    ])

    dataset = COCOSegmentationDataset(
        root_dir=root_dir,
        split=split,
        year=year,
        transform=image_transform,
        target_transform=target_transform_pil
    )
    print(f"COCOSegmentationDataset created with {len(dataset)} samples for {split}{year} split.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if split == 'train' else False
    )
    print(f"DataLoader created for {split}{year}. Number of batches: {len(dataloader)}")

    return dataloader

if __name__ == '__main__':
    COCO_ROOT_DIR = "/path/to/your/coco"

    if COCO_ROOT_DIR == "/path/to/your/coco" or not os.path.exists(COCO_ROOT_DIR):
        print("=" * 70)
        print("WARNING: Please update 'COCO_ROOT_DIR' in the example section with the")
        print("         actual path to your COCO dataset and ensure it exists.")
        print("         Skipping dataloader example.")
        print("=" * 70)
    else:
        print(f"Using COCO_ROOT_DIR: {COCO_ROOT_DIR}")
        try:
            train_dataloader = get_coco_dataloader(
                root_dir=COCO_ROOT_DIR,
                split='train',
                year='2017',
                batch_size=2,
                shuffle=True,
                img_size=(128, 128)
            )

            print(f"\nSuccessfully created train_dataloader.")

            for i, (images, masks) in enumerate(train_dataloader):
                print(f"\nBatch {i + 1}:")
                print(f"  Images shape: {images.shape}, dtype: {images.dtype}")
                print(f"  Masks shape: {masks.shape}, dtype: {masks.dtype}")
                print(f"  Mask unique values: {torch.unique(masks)}")

                if i == 1:
                    break
            print("\nExample dataloader iteration finished.")

        except FileNotFoundError as e:
            print(f"\nERROR: Could not load dataset. FileNotFoundError: {e}")
            print("Please ensure the COCO dataset is correctly placed and paths are correct.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
