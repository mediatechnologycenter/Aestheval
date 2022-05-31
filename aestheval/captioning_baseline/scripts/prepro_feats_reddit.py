"""
Preprocess a raw json dataset into features files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: two folders of features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

from torchvision import transforms as trn

import ipdb ; ipdb.set_trace()
from captioning.utils.resnet_utils import myResnet
import captioning.utils.resnet as resnet

import re
from nltk.tokenize import word_tokenize


class ResizeBigImages(trn.Resize):
    def __init__(self, size):
        super(ResizeBigImages, self).__init__(size)

    def forward(self, im):
        h, w = im.size
        if max(im.size) > self.size:
            if h > w:
                new_h = self.size
                new_w = self.size * w // h
            else:
                new_h = self.size * h // w
                new_w = self.size
            return trn.Resize((new_w,new_h))(im)
        else:
            return im


def remove_URL(text):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "<URL>", text)


def prepare_json(params):
    imgs = []
    exclude = set(string.punctuation)
    for s in ['train', 'validation', 'test']:
        for c in json.load(open(params['input_json']+"/processed_%s.json" % s, 'r')):
            captions = []
            for flc in c['first_level_comments_values']:
                tmp = [t for t in word_tokenize(remove_URL(flc.lower())) if not t in exclude]
                if len(tmp) > 0:
                    captions.append({'tokens': tmp})
            if len(captions) > 0:
                imgs.append({'imgid': c['im_paths'], 'sentences': captions, 'split': s})
    return imgs


def main(params):
    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
    my_resnet = myResnet(net)
    device = 'cuda:1'
    my_resnet.to(device)
    my_resnet.eval()

    imgs = prepare_json(params)    
    N = len(imgs)

    seed(123) # make reproducible

    dir_fc = params['output_dir']+'_fc'
    dir_att = params['output_dir']+'_att'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    preprocess = trn.Compose([
                    ResizeBigImages(512),
                    trn.ToTensor(),
                    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for i,img in enumerate(imgs):
        # load the image
        I = Image.open(os.path.join(params['images_root'], img['imgid'])).convert("RGB")
        I = preprocess(I).to(device)
        with torch.no_grad():
            tmp_fc, tmp_att = my_resnet(I, params['att_size'])
        # write to pkl
        np.save(os.path.join(dir_fc, os.path.basename(img['imgid'].split('.')[0])), tmp_fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(dir_att, os.path.basename(img['imgid'].split('.')[0])), feat=tmp_att.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    print('wrote ', params['output_dir'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
    parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
