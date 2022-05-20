# from aestheval.baselines.model import nima, mlsp
from aestheval.baselines.training.build_models import _build_model
from aestheval.baselines.training.trainer import train, val
from aestheval.baselines.evaluating.evaluate import evaluate
import sys
from aestheval.data.datasets import PCCD, Reddit, AVADataset
import os
import torch
from torch.utils.data import DataLoader

SPLITS = ('train', 'validation', 'test')

def run(dataset_name, method):
    assert dataset_name in ['PCCD', 'Reddit', 'AVADataset']
    assert method in ['nima', 'mlsp', 'vit']
    dataset = {split: eval(dataset_name)(split) for split in SPLITS}

    if method == "vit":
        
        checkpoints_dir = "/media/data-storage/weights/aestheval/"
        name= "test"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        config = {
            "model_name": "vit",
            "verbose": True,
            "this_run_checkpoints": os.path.join(checkpoints_dir, method + '-' + dataset_name + '-' + name),
            "load_existing_model": False,
            "lr": 1e-6,
            "continue_from_epoch": 0,
            "num_epochs": 5, 
            "loss": "mse",
            "batch_log_interval": 250,
            "save_model_every": 3,
            "batch_size": 1
        }
        model = _build_model(config).to(device)

        train_loader = DataLoader(dataset['train'],
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=4)
        valid_loader = DataLoader(dataset['validation'],
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=4)

        train(model, train_loader, valid_loader, config, device)

        test_loader = DataLoader(dataset['test'],
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=4)
        results = evaluate(model, test_loader, config, device)

        torch.save(results, os.path.join(config["this_run_checkpoints"], 'results.pt'))
    
    else:
        eval(method).train(dataset_name, dataset)
        eval(method).evaluate(dataset_name, dataset['test'])