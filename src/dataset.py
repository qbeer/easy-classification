from PIL import Image
import numpy as np
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class CSVDataset(Dataset):
    def __init__(self, transform, df, base_img_path='./rsna_preprocessed/images_as_pngs_512/train_images_processed_512', output_cols=['cancer']):
        assert 'cancer' in output_cols, 'Cancer should be an output column!'
        self.df = df
        counts = self.df['cancer'].value_counts()
        self.weights = self.df['cancer'].apply(lambda x: 1/counts[x]).values
        self.base_img_path = base_img_path
        self.outputs = output_cols
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path, cancer = self.df.iloc[idx][['cleaned_path', 'cancer']]
        img = Image.open(file_path).convert("RGB")

        return {
            "image": self.transform(img),
            "cancer": torch.tensor([cancer])
        }