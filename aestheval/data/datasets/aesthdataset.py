import torch
import cv2
from abc import ABC, abstractmethod
from torchvision import transforms
from pathlib import Path
import os
from PIL import Image

# def valid_image(img, im_path):
#     if img is None or img.size == 0:
#         print("Image {} cannot be read".format(str(im_path)))
#         print(img)
#         raise ValueError(f"Corrupted or missing image with path: {str(im_path)}")

#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

#     img = img.copy()
#     return img

class AestheticsDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, 
        split,
        dataset_path,
        image_dir,
        file_name,
        transform=None,
        load_images: bool = True,
        min_words=0,
        min_info_score=0
        ):
        """
        Args:
            max_size (int, optional): max images size. Defaults to 700.
            transform ([type], optional): [description]. Defaults to None.
            mode (str): Default is "train".
        """
        assert split in ["train", "test", "validation"], "Split must be one of those: 'train', 'test', 'validation'"

        self.load_images = load_images
        self.dataset_path = dataset_path
        self.split = split
        self.image_dir = image_dir
        self.file_name = file_name
        self.transform = transform
        self.min_words = min_words
        self.min_info_score = min_info_score
        if transform is None:
            self.transform = transforms.ToTensor()
        
    def post_dataset_load(self):

        # Assert there's a dataset and ids
        assert self.dataset, "Dataset was loaded incorrectly"
        assert self.dataset_name, "Dataset name is missing"
        assert self.ids, "ids were loaded incorrectly"

        # Filter dataset by words and info score and add dataset name at sample level
        dataset = []
        for data in self.dataset:
            indexed_comments = [(k, x, info_score) for k, x, info_score in zip(data['sentiment'], data['comments'], data['info_scores']) if (len(x.split()) > self.min_words) and (info_score > self.min_info_score)] #Get only the comments with at least `self.min_words` words
            comments = []
            sentiments = {}
            info_scores = []

            for i, (idx, comment, i_score) in enumerate(indexed_comments):
                comments.append(comment)
                info_scores.append(i_score)
                sentiments[i] = data['sentiment'][idx]
            
            if len(comments):
                data['comments'] = comments
                data['sentiment'] = sentiments
                data['info_scores'] = info_scores
                data['dataset_name'] = self.dataset_name
                dataset.append(data)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        data = self.dataset[idx]
        if self.load_images:
            image_file = os.path.join(self.image_dir, data[self.file_name])
            im = Image.open(image_file).convert('RGB')
            # im = valid_image(im, image_file)
            image = self.transform(im)
        else:
            image=None
        return image, data 
