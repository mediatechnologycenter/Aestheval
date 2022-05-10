# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import io
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.utils import pre_caption


class pccd(Dataset):
    def __init__(self,
                 folder: str,
                 ann_root: str,
                 split: str,
                 transform=None,
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
        """
        path = Path(folder)
        datafile = os.path.join(ann_root, "guru_%s.json" % split.lower())
        data = json.load(io.open(datafile), encoding = 'utf-8')
        self.image_folder = os.path.join(path, "images", "full")
        self.textenc_folder = os.path.join(path, "text_emb")

        self.dict_keys =['general_impression', 'subject_of_photo', 'composition',
                         'use_of_camera', 'depth_of_field', 'color_lighting',
                         'focus']
        self.keys = [{k: d[k] for k in self.dict_keys + ['title', 'score']} for d in data]

        self.transform = transform
        self.is_train = True if split == 'TRAIN' else False

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        data = self.keys[ind]
        image_file = os.path.join(self.image_folder, data['title'])
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)
        attr_id = []
        captions = []
        for i, caption in enumerate([data[k] for k in self.dict_keys]):
            if caption:
                captions.append(pre_caption(caption, 30))
                attr_id.append(i)

        text_emb = torch.load(os.path.join(self.textenc_folder, data['title']+'.pth'))

        # Success
        return image, text_emb, attr_id, data['title']