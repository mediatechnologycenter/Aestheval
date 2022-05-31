# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import io
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


dict_keys =['general_impression', 'subject_of_photo', 'composition', 
            'use_of_camera', 'depth_of_field', 'color_lighting', 'focus']

def dict2list(comment_aspects):
    comment_scores = []
    for ca in comment_aspects:
        list2dict = {item['label']: item['score'] for item in ca}
        scores = [list2dict[d] for d in dict_keys]
        comment_scores.append(scores)
    return comment_scores


class Reddit(Dataset):
    def __init__(self,
                 split: str,
                 dataset_path: str = 'data/reddit/',
                 split_path: str = "aestheval/data/datasets/reddit/",
                 transform=None,
                 load_images: bool = True):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            data_path (str): Folder containing images and text files matched by their paths' respective "stem"
        """
        assert split in ["train", "test", "validation"], "Split must be one of those: 'train', 'test', 'validation'"

        self.image_folder = Path(dataset_path)
        self.load_images = load_images
        # Get split

        self.processed=False

        if os.path.exists(Path(dataset_path, f"processed_{split}.json")):
            split_file = Path(dataset_path, f"processed_{split}.json")
            self.processed = True
            with open(split_file, 'r') as f:
                self.dataset = json.load(f)
        else:

            datafile = os.path.join(dataset_path, "reddit_photocritique_image_comments.json")
            with open(datafile, 'r') as f:
                data = json.load(f)
            data = pd.DataFrame(data)
            ids = pd.read_csv(f"{split_path}{split}_ids.csv", header=None, names=['im_paths'])
            data = data[data['im_paths'].isin(ids['im_paths'])]
            for d in data:
                d['im_id'] = d['im_paths'].split('submission_')[1].split('-')[0]

            self.dataset = json.loads(data.to_json(orient='records', indent=1))

        # The order of these attributes it's important to match with the order of scores
        self.aesthetic_attributes = ['general_impression', 'subject_of_photo', 'composition',
                         'use_of_camera', 'depth_of_field', 'color_lighting',
                         'focus']

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()

        self.is_train = True if split == 'TRAIN' else False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        data = self.dataset[ind]
        if self.load_images:
            image_file = os.path.join(self.image_folder, data["im_paths"])
            image = Image.open(image_file).convert('RGB')
            image = self.transform(image)
        else:
            image = None
        data["im_score"] = data['mean_score'] * 10 # x10 to set the scores between 0 and 10
        return image, data
