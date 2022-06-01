import sys
from aestheval.data.datasets import PCCD, Reddit, AVA
import os
import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from scipy import stats
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression
import pandas as pd
import json


MODEL_NAMES = ['vit_base_patch16_224_in21k', 'vit_base_patch16_224', 'vit_deit_small_patch16_224', 'vit_small_patch16_224',
                 'vit_deit_tiny_patch16_224', 'vit_deit_base_patch16_224', 'vit_base_patch32_224', 'vit_large_patch16_224_in21k', 'vit_large_patch16_224']

SPLITS = ('train', 'validation', 'test')
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on ", device)

def get_features(dataset, model):
    all_features = []
    all_labels = []
    dataloader = DataLoader(dataset, batch_size=1)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for im, score in tqdm(dataloader):
            images, labels = im, score
            features = model.forward_features(images.to(device))

            images.detach()
            all_features.append(features.cpu().detach())
            all_labels.append(labels.cpu().detach())

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

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


def run(dataset_name, model_name, root_dir):

    assert model_name in MODEL_NAMES

    if dataset_name == 'PCCD':
        dataset = {split: eval(dataset_name)(split, dataset_path=os.path.join(root_dir, 'PCCD')) for split in SPLITS}
    elif dataset_name == 'Reddit' or dataset_name =='AVA':
        dataset = {split: eval(dataset_name)(split, dataset_path=os.path.join(root_dir, dataset_name.lower())) for split in SPLITS}
    else:
        raise ValueError("Dataset not implemented")

    

    model = timm.create_model(model_name, pretrained=True)

    train_features, train_labels = get_features(dataset['train'], model)
    test_features, test_labels = get_features(dataset['test'], model)


    # Perform logistic regression
    classifier = LinearRegression()
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)

    metrics = get_metrics(test_labels, predictions)
    output_folder = os.path.join('results', dataset_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(os.path.join(output_folder, model_name + '_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)


