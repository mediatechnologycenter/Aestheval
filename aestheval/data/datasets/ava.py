from pathlib import Path
import os
import json
from torchvision import transforms
from PIL import Image
from aestheval.data.datasets.aesthdataset import AestheticsDataset
import pandas as pd
from tqdm import tqdm


path = Path(os.path.dirname(__file__))
ava_files_path = Path(path.parent, 'ava')


class AVA(AestheticsDataset):
    def __init__(
        self,
        split,
        dataset_path = 'data/ava/',
        transform=None,
        load_images: bool = True,
        min_words = 0,
        informativeness=False,
        min_info_score=0
    ):

        image_dir = Path(dataset_path, "images")
        self.dataset_name = 'ava'

        AestheticsDataset.__init__(self, 
            split=split,
            dataset_path=dataset_path,
            image_dir=image_dir,
            file_name='im_name',
            transform=transform,
            load_images=load_images,
            min_words=min_words,
            min_info_score=min_info_score)

        score_file=os.path.join(ava_files_path, "dpchallenge_id_score.json")
        db_file=os.path.join(ava_files_path,"uncorrupted_images.json")
        metadata_file=os.path.join(ava_files_path,"AVA_data_official_test.csv")

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
            self.ids = [data['im_id'] for data in self.dataset]
        else:
            with open(score_file, "r") as f:
                self.score_map = json.load(f)

            with open(db_file, "r") as f:
                self.image_key_map = json.load(f)

            ids = pd.read_csv(metadata_file)           
            train_ids = ids[ids["set"] == "training"]
            validation_ids = ids[ids["set"] == "validation"]
            test_ids = ids[ids["set"] == "test"]

            self.train_image_names = list(train_ids["image_name"])
            self.validation_image_names = list(validation_ids["image_name"])
            self.test_image_names = list(test_ids["image_name"])

            self.labels = {}
            for image_name, score, challenge_id in ids[
                ["image_name", "MOS", "challenge_id"]
            ].itertuples(index=False):
                self.labels[image_name] = [score, challenge_id]

            self.preprocess_data()
        
        # self.post_dataset_load()
        

    def preprocess_data(self):
        if self.split == "train":
            print("Loading ava train set")
            im_list = self.train_image_names
        elif self.split == "validation":
            print("Loading ava val set")
            im_list = self.validation_image_names
        else:
            print("Loading ava test set")
            im_list = self.test_image_names

        self.dataset = []
        self.ids = []
        discarded_images = []
        for _, im_name in enumerate(im_list):

            im_id = Path(im_name).stem

            if not (im_id in self.image_key_map):
                discarded_images.append(im_id)
                continue

            self.ids.append(im_id)
            self.dataset.append(
                {
                    "im_id": im_id,
                    "im_score": self.score_map[im_id],
                    "query": self.labels[im_name][1],
                    "im_name": im_name,
                }
            )
        imgs_not_found = self.add_comments()
        print(f"Discarded {len(discarded_images) + len(imgs_not_found)} out of {len(im_list)} images from AVA {self.split} set")
    
    def add_comments(self):
        AVA_comments = {}

        imgs_not_found = []
        with open(Path(self.dataset_path, 'ava_comments.txt'), 'r', encoding = 'utf-8') as f:
            for line in f.readlines():
                elements = [elem.strip() for elem in line.strip('\n').split('#')]
                
                id, captions = elements[1], elements[2:]       

                if not os.path.exists(Path(self.image_dir, id + '.jpg')):
                    imgs_not_found.append(id)
                if len(captions):
                    AVA_comments[id] = captions
        
        dataset = []
        for data in self.dataset:
            if data["im_id"] in AVA_comments:
                data['comments'] = AVA_comments[data["im_id"]]
                dataset.append(data)
        
        self.dataset = dataset
        
        return imgs_not_found
    
