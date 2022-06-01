from aestheval.baselines.model import nima#, mlsp
import sys
from aestheval.data.datasets import PCCD, Reddit, AVA
import os
import torch
from torch.utils.data import DataLoader

SPLITS = ('train', 'validation', 'test')


def run(dataset_name, method, root_dir, evaluate):
    assert dataset_name in ['PCCD', 'Reddit', 'AVA']
    assert method in ['nima', 'mlsp']
    
    if dataset_name == 'PCCD':
        dataset = {split: eval(dataset_name)(split, dataset_path=os.path.join(root_dir, 'PCCD')) for split in SPLITS}
    elif dataset_name == 'Reddit' or dataset_name =='AVA':
        dataset = {split: eval(dataset_name)(split, dataset_path=os.path.join(root_dir, dataset_name.lower())) for split in SPLITS}
    else:
        raise ValueError("Dataset not implemented")

    if not evaluate:
        eval(method).train(dataset_name, dataset)
    eval(method).evaluate(dataset_name, dataset['test'])