import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import numpy as np
import os

class COCOSegmentationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, processor, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Load image
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.root_dir, img_info['file_name'])).convert('RGB')
        
        # Load mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            mask += self.coco.annToMask(ann)
        mask = mask > 0
        mask = Image.fromarray(mask)

        # Get bounding box
        boxes = []
        for ann in anns:
            boxes.append(ann['bbox'])  # [x, y, width, height]
        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2:] += boxes[:, :2]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)

        original_size = tuple(image.size)[::-1]  # (height, width)

        # Apply processor
        inputs = self.processor(image, original_size, boxes[0].tolist() if len(boxes) > 0 else None)
        
        # Add ground truth mask
        inputs["ground_truth_mask"] = torch.from_numpy(np.array(mask))

        if self.transform:
            inputs = self.transform(inputs)

        return inputs

def collate_fn(batch):
    return list(batch)