"""
Dataset y DataLoader en PyTorch con Data Augmentation para emulacion de condiciones UAV.
Incluye precarga en memoria RAM (73 MB) para aceleracion masiva de entrenamiento.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class UAVShipDataset(Dataset):
    """
    Dataset optimizado para imagenes satelitales 80x80 px RGB.
    Soporta precarga directa en RAM para eliminar cuellos de botella de disco.
    """
    def __init__(self, metadata_df, transform=None, preload=True):
        self.df = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.preload = preload
        self.images = None
        self.labels = self.df['label'].values.astype(np.float32)
        
        if self.preload:
            img_list = []
            for _, row in self.df.iterrows():
                with Image.open(row['filepath']) as img:
                    img_list.append(np.array(img.convert('RGB'), dtype=np.uint8))
            self.images = np.stack(img_list, axis=0) # (N, 80, 80, 3)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.images is not None:
            image = Image.fromarray(self.images[idx])
        else:
            with Image.open(self.df.iloc[idx]['filepath']) as img:
                image = img.convert('RGB')
                
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms():
    """
    Retorna las transformaciones de entrenamiento (Data Augmentation UAV) y validacion.
    """
    mean = [0.3821, 0.4055, 0.3545]
    std = [0.1450, 0.1088, 0.0999]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform
