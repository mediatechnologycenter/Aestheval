"""
file - model.py
Implements the aesthemic model and emd loss used in paper.

Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os
import glob
import torch
import torch.nn as nn

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from scipy import stats
from tqdm import tqdm

from torchvision.models import vgg16
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sklearn.metrics as sm


"""Neural IMage Assessment model by Google"""


class Model(nn.Module):

    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        base_model = vgg16(pretrained=True)
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def single_emd_loss(p, q, r=2):
    """
    Earth Mover's Distance of one sample

    Args:
        p: true distribution of shape num_classes × 1
        q: estimated distribution of shape num_classes × 1
        r: norm parameter
    """
    assert p.shape == q.shape, "Length of the two distribution must be the same"
    length = p.shape[0]
    emd_loss = 0.0
    for i in range(1, length + 1):
        emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
    return (emd_loss / length) ** (1. / r)


class Loss(nn.Module):
    def forward(self, p, q, r=2):
        """
        Earth Mover's Distance on a batch

        Args:
            p: true distribution of shape mini_batch_size × num_classes × 1
            q: estimated distribution of shape mini_batch_size × num_classes × 1
            r: norm parameters
        """
        assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
        mini_batch_size = p.shape[0]
        loss_vector = []
        for i in range(mini_batch_size):
            loss_vector.append(single_emd_loss(p[i], q[i], r=r))
        return sum(loss_vector) / mini_batch_size


def collate_fn(batch):
    def from_dict_to_tensor(dictionary):
        values = list(dictionary.values())
        values = torch.tensor([list(v.values()) for v in values if v is not None])
        return values.mean(0)

    images = torch.stack([sample[0] for sample in batch])
    labels = torch.stack([from_dict_to_tensor(sample[1]['sentiment']) for sample in batch])
    return images, labels


def train(dataset_name, dataset, batch_size=64, epochs=100, early_stopping_patience=5):
    ckpt_path = 'ckpts/NIMA/%s' % dataset_name

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running training on {device}")
    writer = SummaryWriter()

    dataset['train'].transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    dataset['validation'].transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    num_classes = 3
    model = Model(num_classes=num_classes)
    model = model.to(device)

    conv_base_lr = 3e-7
    dense_lr = 3e-4
    criterion = Loss()
    optimizer = optim.SGD([
        {'params': model.features.parameters(), 'lr': conv_base_lr},
        {'params': model.classifier.parameters(), 'lr': dense_lr}],
        momentum=0.9
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    param_num = 0
    for param in model.parameters():
        if param.requires_grad:
            param_num += param.numel()
    print('Trainable params: %.2f million' % (param_num / 1e6))

    # dataset['train'] = torch.utils.data.Subset(dataset['train'], range(10))
    # dataset['validation'] = torch.utils.data.Subset(dataset['validation'], range(10))
    train_loader = DataLoader(dataset['train'],
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset['validation'],
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)

    # for early stopping
    count = 0
    init_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        batch_losses = []
        for i, data in enumerate(tqdm(train_loader)):
            images = data[0].to(device)
            labels = data[1].to(device).float()
            outputs = model(images)
            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch: %d/%d | Step: %d/%d | Training EMD loss: %.4f' % (epoch + 1, epochs, i + 1, len(dataset['train']) // batch_size + 1, loss.item()))
            writer.add_scalar('batch train loss', loss.item(), i + epoch * (len(dataset['train']) // batch_size + 1))

        avg_loss = sum(batch_losses) / (len(dataset['train']) // batch_size + 1)
        train_losses.append(avg_loss)
        print('Epoch %d mean training EMD loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()

        # do validation after each epoch
        batch_val_losses = []
        print("Running validation...")
        for data in tqdm(val_loader):
            images = data[0].to(device)
            labels = data[1].to(device).float()
            with torch.no_grad():
                outputs = model(images)
            val_loss = criterion(labels, outputs)
            batch_val_losses.append(val_loss.item())
        avg_val_loss = sum(batch_val_losses) / (len(dataset['validation']) // batch_size + 1)
        val_losses.append(avg_val_loss)
        print('Epoch %d completed. Mean EMD loss on val set: %.4f.' % (epoch + 1, avg_val_loss))
        writer.add_scalars('epoch losses', {'epoch train loss': avg_loss, 'epoch val loss': avg_val_loss}, epoch + 1)

        # Use early stopping to monitor training
        if avg_val_loss < init_val_loss:
            init_val_loss = avg_val_loss
            # save model weights if val loss decreases
            print('Saving model...')
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            torch.save(model.state_dict(), os.path.join(ckpt_path, 'epoch-%d.pth' % (epoch + 1)))
            print('Done.\n')
            # reset count
            count = 0
        elif avg_val_loss >= init_val_loss:
            count += 1
            if count == early_stopping_patience:
                print('Val EMD loss has not decreased in %d epochs. Training terminated.' % early_stopping_patience)
                break

    print('Training completed.')

@torch.no_grad()
def evaluate(dataset_name, dataset, batch_size=64):
    import re

    def extract_number(f):
        s = re.findall(r'\d+', f)
        return (int(s[0]) if s else -1, f)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")
    ckpt_path = 'ckpts/NIMA/%s' % dataset_name
    filenames = glob.glob(ckpt_path+"/*.pth")

    num_classes = 3
    model = Model(num_classes=num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(max(filenames, key=extract_number))) # we load the weights of the last checkpoint
    model.eval()

    dataset.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])])

    # dataset = torch.utils.data.Subset(dataset, range(10))
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             collate_fn=collate_fn)

    predictions = []
    groundtruths = []
    for data in tqdm(test_loader):
        images = data[0].to(device)
        labels = data[1].float()
        output = model(images)
        predictions.append(output.cpu())
        groundtruths.append(labels)

    gt_distr = torch.cat(groundtruths)
    pred_distr = torch.cat(predictions)

    # Coherently with Eq. 1, we expect pos, neu, neg. Tensor's
    # columns are neg, neu, pos, thus we flip the columns
    gt = (3 - torch.sum(gt_distr.fliplr() * torch.arange(1,4), 1)) / 2
    pred = (3 - torch.sum(pred_distr.fliplr() * torch.arange(1,4), 1)) / 2

    srcc = stats.spearmanr(gt, pred)
    print("SRCC =", srcc)
    mse = round(sm.mean_squared_error(gt, pred), 4)
    print("MSE =", mse)
    lcc = stats.pearsonr(gt, pred)
    print("LCC =", lcc)

    with open(ckpt_path+"/results.txt", 'w') as f:
        f.write("SRCC = {}\n".format(srcc))
        f.write("MSE = {}\n".format(mse))
        f.write("LCC = {}\n".format(lcc))

    torch.save({'gt_distr': gt_distr, 'pred_distr': pred_distr,
                'gt': gt_distr, 'pred': pred_distr,
                }, ckpt_path+"/predictions.pth")