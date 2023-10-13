import argparse
from copy import copy

import torch
import torchvision
from torch import nn

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms.transforms import RandomApply
from torch.utils.data import WeightedRandomSampler

from src.efficient_net import efficient_net
from src.backbone_model import BackboneClassifier
from torchvision.datasets import ImageFolder

from pytorch_lightning import seed_everything
from collections import Counter

def get_class_weights(targets, n_classes):
    class_dist = Counter(targets)
    class_weights = [class_dist[i] / np.sum([class_dist[j] for j in range(n_classes)]) for i in range(n_classes)]
    
    print('N CLASSES')
    print(n_classes)
    
    print('CLASS WEIGHTS')
    print(class_weights)
    
    return class_weights

def check_corrupted(path):
    from PIL import Image
    try:
        Image.open(path)
        return True
    except:
        return False

seed_everything(42, workers=True)

def train(args):
    backbone, transforms, val_transforms = efficient_net(args.model_identifier.split('_')[0])

    ds = ImageFolder('./data', transform=transforms, target_transform=None, is_valid_file=check_corrupted)
    n_classes = len(ds.classes)
        
    model = BackboneClassifier(backbone, n_classes, learning_rate=1e-4)
    
    train_ds, val_ds = torch.utils.data.random_split(ds, [int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)])
    val_ds.dataset.transform = val_transforms
    
    train_ds.dataset.targets = np.array(train_ds.dataset.targets)
    
    class_weights = get_class_weights(train_ds.dataset.targets, n_classes=n_classes)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=5,
        pin_memory=False)
    
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=5,
        pin_memory=False
    )

    wandb_logger = WandbLogger(
        name="classification_test_run", project='classification', entity='elte-ai4covid', 
        save_dir='./lightning_logs/')

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath='./lightning_logs/classification/',
        save_top_k=3,
        mode='min',
        filename=f'{args.model_identifier}-' + '{epoch:02d}-{valid_loss:.3f}')

    acc_checkpoint_callback = ModelCheckpoint(
        monitor='valid_accuracy',
        dirpath='./lightning_logs/classification/',
        save_top_k=3,
        mode='max',
        filename=f'{args.model_identifier}-' + '{epoch:02d}-{valid_accuracy:.3f}')
    
    trainer = pl.Trainer(max_epochs=args.epochs,
                         devices=1,
                         accelerator='gpu',
                         logger=wandb_logger,
                         val_check_interval=1.0,
                         min_epochs=args.epochs,
                         callbacks=[
                             checkpoint_callback,
                             acc_checkpoint_callback
                         ],
                         deterministic=True)

    trainer.fit(model, train_dl, val_dl)
    trainer.test(dataloaders=val_non_balanced_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10, required=False)
    parser.add_argument(
        '--model_identifier', choices=['b8_1000', 'b8_672', 'b8_300', 'b3_300', 'b3_512'],
        required=False, default='b3_512')

    args = parser.parse_args()
    train(args)
