# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import io
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class PCCD(Dataset):
    def __init__(self,
                 split: str,
                 data_path: str = "../data/PCCD",
                 split_path: str = "../data_splits/PCCD",
                 transform=None,
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
        """
        data_path = Path(data_path)
        datafile = os.path.join(data_path, "guru.json")
        split_file = os.path.join(split_path, f"{split.lower()}_ids.csv")
        data = json.load(io.open(datafile), encoding = 'utf-8')
        split_ids = pd.read_csv(split_file, header=None)
        
        #TODO: select data by ids
        data = data

        self.image_folder = os.path.join(data_path, "images", "full")
        
        # The order of these attributes it's important to match with the order of scores
        self.attributes = ['general_impression', 'subject_of_photo', 'composition',
                         'use_of_camera', 'depth_of_field', 'color_lighting',
                         'focus']
        self.selected_keys = self.attributes + ['description','title', 'score', 'category']
        
        self.dataset = [{k: d[k] for k in self.selected_keys} for d in data]

        self.transform = transform
        self.is_train = True if split.lower() == 'train' else False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        data = self.dataset[ind]
        data['im_name'] = data.pop('title') #Rename
        image_file = os.path.join(self.image_folder, data['im_name'])
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)

        return image, data