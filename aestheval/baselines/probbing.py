import sys
from aestheval.data.datasets import PCCD, Reddit, AVA
import os
import torch
from torch.utils.data import DataLoader
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from scipy import stats
import sklearn.metrics as sm
from sklearn.linear_model import SGDRegressor
import pandas as pd
import json
import numpy as np

MODEL_NAMES = ['vit_base_patch16_224_in21k', 'vit_base_patch16_224', 'vit_deit_small_patch16_224', 'vit_small_patch16_224',
                 'vit_deit_tiny_patch16_224', 'vit_deit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224_in21k', 'vit_large_patch16_224']

SPLITS = ('train', 'validation', 'test')
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on ", device)


def collate_fn_sentiment_score(batch):
    images = torch.stack([sample[0] for sample in batch])
    labels = torch.stack([torch.tensor(sample[1]['mean_score'] * 10) for sample in batch]) #x10 to scale it to [0,10]
    return images, labels

def collate_fn_original_ava(batch):
    images = torch.stack([sample[0] for sample in batch])
    labels = torch.stack([torch.tensor(sample[1]['im_score']) for sample in batch]) 
    return images, labels

def collate_fn_original_pccd(batch):
    images = torch.stack([sample[0] for sample in batch])
    # print(batch)
    labels = torch.stack([torch.tensor([float(x) for x in sample[1]['score'] if x]).mean() for sample in batch]) 
    return images, labels

def get_features(dataset, model, collate_fn):
    all_features = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels = data
            features = model.forward_features(images.to(device))

            images.detach()
            all_features.append(features.detach().cpu().numpy())
            all_labels.append(labels)

    return np.concatenate(all_features), torch.cat(all_labels).numpy()

def get_metrics(labeller_score_list: torch.Tensor, network_score_list: torch.Tensor, verbose=True) -> dict:
    srcc = stats.spearmanr(labeller_score_list, network_score_list)
    print("SRCC =", srcc)
    mse = round(sm.mean_squared_error(labeller_score_list, network_score_list), 4)
    print("MSE =", mse)
    lcc = stats.pearsonr(labeller_score_list, network_score_list)
    print("LCC =", lcc)
    acc = calc_accuracy(labeller_score_list=labeller_score_list, network_score_list=network_score_list)
    print("Accuracy th=5, delta=0 ->", acc)

    return {"SRCC": srcc[0],
            "MSE": mse,
            "LCC": lcc[0],
            "accuracy": acc}

def calc_accuracy(labeller_score_list: list, network_score_list: list, th: float = 5.0, delta:float = 0.0):
    # Transform to series
    labeller_score_list = pd.Series(labeller_score_list)
    network_score_list = pd.Series(network_score_list)
    # Get positive and negative labels, filtering those scores in delta range of the grounf truths
    top = labeller_score_list > th + delta
    bottom = labeller_score_list < th - delta
    labeller_score_list = labeller_score_list[top | bottom] > th
    network_score_list = network_score_list[top | bottom] > th
    
    return sm.accuracy_score(labeller_score_list, network_score_list)


def run(dataset_name, model_name, root_dir, scoring='original'):

    assert model_name in MODEL_NAMES

    model = timm.create_model(model_name, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    if dataset_name == 'PCCD':
        dataset = {split: eval(dataset_name)(split, dataset_path=os.path.join(root_dir, 'PCCD'), transform=transform) for split in SPLITS}
        colalte_fn = collate_fn_original_pccd
    elif dataset_name == 'Reddit' or dataset_name =='AVA':
        dataset = {split: eval(dataset_name)(split, dataset_path=os.path.join(root_dir, dataset_name.lower()), transform=transform) for split in SPLITS}
        colalte_fn = collate_fn_original_ava
    else:
        raise ValueError("Dataset not implemented")

    if scoring == 'sentiment' or dataset_name == 'Reddit':
        colalte_fn = collate_fn_sentiment_score

    train_features, train_labels = get_features(dataset['train'], model, colalte_fn)
    test_features, test_labels = get_features(dataset['test'], model, colalte_fn)


    # Perform logistic regression
    print("Fitting linear regressor...")
    classifier = SGDRegressor()
    print(train_features.shape, train_labels.shape)
    print(type(train_features), type(train_labels))
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)

    metrics = get_metrics(test_labels, predictions)
    output_folder = os.path.join('results', dataset_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(os.path.join(output_folder, f"{model_name}_{scoring}_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)


