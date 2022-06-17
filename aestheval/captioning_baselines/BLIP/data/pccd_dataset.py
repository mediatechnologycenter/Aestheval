import os
import re
import json
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption


def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "<URL>", text)


def prepare_json(image_root, split):
    imgs = []
    for i, c in enumerate(json.load(open(image_root+"/processed_%s.json" % split, 'r'))):
        captions = []
        for k in ['general_impression', 'subject_of_photo', 'composition',
                  'use_of_camera', 'depth_of_field', 'color_lighting', 'focus',
                  'description']:
            captions.append(remove_URL(c[k]))
        imgs.append({'image_id': str(i), 'image': c['im_name'], 'caption': captions})
    return imgs


class pccd_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''

        self.annotation = prepare_json(image_root, 'train')
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, 'images/full', ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'][np.random.randint(len(ann['caption']))], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
    
class pccd_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        
        self.annotation = prepare_json(image_root, split)
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, 'images/full', ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        img_id = ann['image_id']
        
        return image, int(img_id)