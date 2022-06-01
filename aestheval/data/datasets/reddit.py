# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import io
import json
import torch
from PIL import Image
from aestheval.data.datasets.aesthdataset import AestheticsDataset
from torchvision import transforms
import pandas as pd


dict_keys =['general_impression', 'subject_of_photo', 'composition', 
            'use_of_camera', 'depth_of_field', 'color_lighting', 'focus']

path = Path(os.path.dirname(__file__))
reddit_files_path = Path(path.parent, 'reddit')

def dict2list(comment_aspects):
    comment_scores = []
    for ca in comment_aspects:
        list2dict = {item['label']: item['score'] for item in ca}
        scores = [list2dict[d] for d in dict_keys]
        comment_scores.append(scores)
    return comment_scores


class Reddit(AestheticsDataset):
    def __init__(self,
                 split: str,
                 dataset_path: str = 'data/reddit/',
                 transform=None,
                 load_images: bool = True):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            data_path (str): Folder containing images and text files matched by their paths' respective "stem"
        """
        assert split in ["train", "test", "validation"], "Split must be one of those: 'train', 'test', 'validation'"

        image_dir = dataset_path
        
        AestheticsDataset.__init__(self, 
            split,
            dataset_path,
            image_dir,
            transform,
            load_images)

        self.processed=False

        if os.path.exists(Path(dataset_path, f"processed_{split}.json")):
            split_file = Path(dataset_path, f"processed_{split}.json")
            self.processed = True
            with open(split_file, 'r') as f:
                self.dataset = json.load(f)
            for data in self.dataset:
                data["im_score"] = data['mean_score'] * 10 # x10 to set the scores between 0 and 10
        else:

            datafile = os.path.join(dataset_path, "reddit_photocritique_image_comments.json")
            with open(datafile, 'r') as f:
                data = json.load(f)
            data = pd.DataFrame(data)
            ids = pd.read_csv(Path(reddit_files_path, f"{split}_ids.csv"), header=None, names=['im_paths'])
            data = data[data['im_paths'].isin(ids['im_paths'])]
            data['im_id'] = data['im_paths'].apply(lambda x: x.split('submission_')[1].split('-')[0])                

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
            image_file = os.path.join(self.image_dir, data["im_paths"])
            image = Image.open(image_file).convert('RGB')
            image = self.transform(image)
        else:
            image = None
        return image, data
