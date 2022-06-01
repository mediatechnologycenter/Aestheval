import torch
import cv2
from abc import ABC, abstractmethod
from torchvision import transforms
from pathlib import Path
import os
from PIL import Image

def valid_image(img, im_path):
    if img is None or img.size == 0:
        print("Image {} cannot be read".format(str(im_path)))
        print(img)
        raise ValueError(f"Corrupted or missing image with path: {str(im_path)}")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = img.copy()
    return img

class AestheticsDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, 
        split,
        dataset_path,
        image_dir,
        transform=None,
        load_images: bool = True,
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
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()
        
        self.dataset = None
        self.ids = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        data = self.dataset[idx]
        if self.load_images:
            image_file = os.path.join(self.image_dir, data['im_name'])
            im = cv2.imread(image_file)
            im = valid_image(im, image_file)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            image = self.transform(im)
        else:
            image=None
        return image, data['mean_score'] * 10 #x10 to scale it to [0,10]
