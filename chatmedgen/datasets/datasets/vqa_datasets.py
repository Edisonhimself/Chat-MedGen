import torch
from PIL import Image
import os

from chatmedgen.datasets.datasets.base_dataset import BaseDataset




class TestEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor):
        self.loaded_data = loaded_data
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]

        image_path = ann['image_path']
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        prompt = ann["prompt"]

        return image, prompt

