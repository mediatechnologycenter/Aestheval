# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import json
from torchvision import transforms
from PIL import Image
from aestheval.data.datasets.aesthdataset import AestheticsDataset
import ast

path = Path(os.path.dirname(__file__))
pccd_files_path = Path(path.parent, 'PCCD')

class PCCD(AestheticsDataset):
    def __init__(self,
                 split: str,
                 dataset_path: str = "data/PCCD",
                 transform=None,
                 load_images: bool = True,
                 min_words=0,
                 informativeness=False,
                 min_info_score=0
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
        """
        
        image_dir = os.path.join(dataset_path, "images", "full")
        self.dataset_name = 'pccd'

        AestheticsDataset.__init__(self, 
            split=split,
            dataset_path=dataset_path,
            image_dir=image_dir,
            file_name='im_name',
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
        else:
            split_file = os.path.join(pccd_files_path, f"guru_{split.lower()}.json")

        with open(split_file, 'r') as f:
            data = json.load(f)

        
        
        # The order of these attributes it's important to match with the order of scores
        self.attributes = ['general_impression', 'subject_of_photo', 'composition',
                         'use_of_camera', 'depth_of_field', 'color_lighting',
                         'focus']
        
        # To take into account the change of variable name when processing
        if self.processed:
            var_name = 'im_name'
        else:
            var_name = 'title'                 
        self.selected_keys = self.attributes + ['description',var_name, 'score', 'category']


        #If sentiment is already in data
        if "sentiment" in data[0].keys():
            self.selected_keys = self.selected_keys + ["sentiment", 'mean_score', 'stdev_score','number_of_scores', 'info_scores']
        
        self.ids = []
        self.dataset = []
        for d in data:
            dic = {k: d[k] for k in self.selected_keys}
            dic['comments'] = [d[k] for k in self.attributes]
            scores = []
            for score in d['score']:
                try:
                    scores.append(ast.literal_eval(score))
                except:
                    # print("Malformed score, adding None")
                    scores.append(None)
            dic['score'] = scores
            if not self.processed:
                dic['im_name'] = dic.pop('title') # Rename for readibility
            self.ids.append(dic['im_name'])
            self.dataset.append(dic)
                
        self.is_train = True if split.lower() == 'train' else False

        # self.post_dataset_load()