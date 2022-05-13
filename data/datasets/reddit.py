# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import io
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


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
                 folder: str,
                 ann_root: str,
                 split: str,
                 transform=None):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
        """
        datafile = os.path.join(ann_root, "reddit_photocritique_image_comments_%s.json" % split.lower())
        data = json.load(io.open(datafile), encoding = 'utf-8')
        self.image_folder = Path(folder)
        self.im_paths = data['im_paths']
        self.comments = data['first_level_comments_values']
        self.aspects = [dict2list(s) for s in data['aspect_prediction']]
        self.transform = transform
        self.is_train = True if split == 'TRAIN' else False

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, ind):
        image_file = os.path.join(self.image_folder, self.im_paths[ind])
        image = Image.open(image_file).convert('RGB')
        image = self.transform(image)
        aspects = self.aspects[ind]
        comments = self.comments[ind]
        return image, comments, aspects, self.im_paths[ind]