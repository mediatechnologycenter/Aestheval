from aestheval.baselines.model import nima, mlsp
import sys
from aestheval.data.datasets import PCCD, Reddit, AVADataset


SPLITS = ('train', 'validation', 'test')


def run(dataset_name, method):
    assert dataset_name in ['PCCD', 'Reddit', 'AVADataset']
    assert method in ['nima', 'mlsp', 'vit']
    dataset = {split: eval(dataset_name)(split) for split in SPLITS}
    eval(method).train(dataset_name, dataset)
    eval(method).evaluate(dataset_name, dataset['test'])