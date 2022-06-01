# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import json
from torchvision import transforms
from PIL import Image
from aestheval.data.datasets.aesthdataset import AestheticsDataset

class DPC(AestheticsDataset):
    def __init__(self,
                 split: str,
                 dataset_path: str = "data/dpc/dpc.json",
                 images_path: str = 'data/ava/images/',
                 transform=None,
                 load_images: bool = True
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
        """

        AestheticsDataset.__init__(self, 
            split,
            dataset_path,
            images_path,
            transform,
            load_images)

        self.processed=False

        
        self.attributes = ['color_lighting', 'composition', 'depth_and_focus', 'impression_and_subject', 'use_of_camera']
        
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)

        self.dataset = []
        self.ids=[]
        for data in dataset:
            if os.path.exists(os.path.join(self.image_dir, data['im_name'])):
                self.ids.append(data['im_name'])
                self.dataset.append(data)
                
        self.is_train = True if split.lower() == 'train' else False