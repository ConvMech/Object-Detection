import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import BBoxDataset
from trainer import Trainer


def prepare_dataset():
    df = pd.read_csv('../annotation.csv')

    class_to_idx = {label: idx for idx, label in enumerate(df['label'].unique())}
    df['filename'] = df['filename'].apply(lambda x: os.path.join('..', x))
    images = df['filename'].tolist()

    groupby_image = df.groupby('filename')
    bboxes, labels = [], []
    for img in images:
        rows = groupby_image.get_group(img)
        # Bouding Box
        x_min = rows['x_min'].values
        y_min = rows['y_min'].values
        x_max = rows['x_max'].values
        y_max = rows['y_max'].values
        bbox = np.stack([x_min, y_min, x_max, y_max]).reshape(-1, 4)
        bbox = torch.from_numpy(bbox)
        bboxes.append(bbox)
        # Labels
        label = rows['label'].map(class_to_idx).values
        label = torch.from_numpy(label).view(-1)
        labels.append(label)

    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])

    return images, bboxes, labels


def bbox_collate_fn(batch):
    images = []
    targets = []
    for sample in batch:
        image, target = sample
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets


if __name__ == '__main__':
    images, bboxes, labels = prepare_dataset()
    train_images, valid_images, train_bboxes, valid_bboxes, \
        train_labels, valid_labels = train_test_split(images, bboxes, labels)
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])
    dtrain = BBoxDataset(train_images, train_bboxes, train_labels, transform)
    dvalid = BBoxDataset(valid_images, valid_bboxes, valid_labels, transform)

    train_loader = DataLoader(dtrain, batch_size=3, drop_last=True,
                              collate_fn=bbox_collate_fn)

    valid_loader = DataLoader(dvalid, batch_size=3,
                              collate_fn=bbox_collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3  # 1 class (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=1e-4)

    trainer = Trainer(model, optimizer)

    num_epochs = 10

    for epoch in range(1, num_epochs+1):
        train_loss = trainer.epoch_train(train_loader)
        valid_loss = trainer.epoch_eval(valid_loader)
        
        print(f'EPOCH: [{epoch}/{num_epochs}]')
        print(f'TRAIN_LOSS: {train_loss}, VALID_LOSS: {valid_loss}')