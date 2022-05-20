# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
import os
import json
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class DPC(Dataset):
    def __init__(self,
                 split: str,
                 data_path: str = "data/dpc/dpc.json",
                 images_path: str = '/media/data-storage/datasets/ava/images/',
                 transform=None,
                 load_images: bool = True
                 ):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
        """

        self.load_images = load_images
        data_path = Path(data_path)
        self.images_path = images_path
        self.processed=False

        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.attributes = ['color_lighting', 'composition', 'depth_and_focus', 'impression_and_subject', 'use_of_camera']
                
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.is_train = True if split.lower() == 'train' else False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        data = self.data[ind]

        if self.load_images:
            image_file = os.path.join(self.images_path, data['im_id'])
            image = Image.open(image_file).convert('RGB')
            image = self.transform(image)
        else:
            image=None

        return image, data