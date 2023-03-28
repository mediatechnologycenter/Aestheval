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
                 load_images: bool = True,
                 min_words=0, 
                 informativeness=False,
                 min_info_score=0):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            data_path (str): Folder containing images and text files matched by their paths' respective "stem"
        """
        assert split in ["train", "test", "validation"], "Split must be one of those: 'train', 'test', 'validation'"

        image_dir = dataset_path
        self.dataset_name = 'reddit'

        AestheticsDataset.__init__(self, 
            split=split,
            dataset_path=dataset_path,
            image_dir=image_dir,
            file_name='im_paths',
            transform=transform,
            load_images=load_images,
            min_words=min_words,
            min_info_score=min_info_score)

        self.processed=False

        processed_file_name = 'processed_'
        if informativeness:
            processed_file_name = 'processed_info_'

        print("Using path: ", Path(self.dataset_path, f"{processed_file_name}{split}.json"))

        if os.path.exists(Path(self.dataset_path, f"{processed_file_name}{split}.json")):
            split_file = Path(self.dataset_path, f"{processed_file_name}{split}.json")
            self.processed = True
            with open(split_file, 'r') as f:
                self.dataset = json.load(f)
            
            self.ids = []
            for data in self.dataset:
                data["im_score"] = data['mean_score'] * 10 # x10 to set the scores between 0 and 10
                data['comments']= data['first_level_comments_values']
                self.ids.append(data['im_id'])
        else:

            datafile = os.path.join(dataset_path, "reddit_photocritique_image_comments.json")
            with open(datafile, 'r') as f:
                data = json.load(f)
            data = pd.DataFrame(data)
            ids = pd.read_csv(Path(reddit_files_path, f"{split}_ids.csv"), header=None, names=['im_id'])
            data['im_id'] = data['im_paths'].apply(lambda x: x.split('/')[1].split('-')[0])
            data = data[data['im_id'].isin(ids['im_id'])]                
            self.ids = data['im_id'].tolist()
            self.dataset = json.loads(data.to_json(orient='records', indent=1))

        # The order of these attributes it's important to match with the order of scores
        self.aesthetic_attributes = ['general_impression', 'subject_of_photo', 'composition',
                         'use_of_camera', 'depth_of_field', 'color_lighting',
                         'focus']

        self.is_train = True if split == 'TRAIN' else False

        # self.post_dataset_load()