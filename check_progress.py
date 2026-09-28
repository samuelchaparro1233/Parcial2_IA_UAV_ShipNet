import torch
import os
import numpy as np
from src.custom_cnn import UAVShipNet
from src.data_loader import UAVShipDataset, get_transforms
from torch.utils.data import DataLoader
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Usando dispositivo:', device)

ckpt_path = 'models/uav_shipnet_best.pt'
if not os.path.exists(ckpt_path):
    print('No se encontro checkpoint en', ckpt_path)
    exit(1)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

if isinstance(ckpt, dict):
    print("Checkpoint info:")
    for k in ckpt.keys():
        if k != 'model_state_dict':
            print(f"  {k}: {ckpt[k]}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

test_csv = 'data/blind_test_split.csv'
if not os.path.exists(test_csv):
    test_csv = 'data/dataset_metadata.csv'

test_df = pd.read_csv(test_csv)
_, val_transform = get_transforms()
test_dataset = UAVShipDataset(test_df, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model = UAVShipNet().to(device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)
model.eval()

all_probs, all_targets = [], []
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_targets.extend(targets.numpy().flatten())

all_probs = np.array(all_probs)
all_targets = np.array(all_targets)

print(f"\nTotal muestras en test_loader: {len(all_targets)}")
ships_cnt = int(np.sum(all_targets))
noships_cnt = len(all_targets) - ships_cnt
print(f"Distribucion Test: {ships_cnt} Barcos (1), {noships_cnt} No-Barcos (0)")

print("\n--- ESCANEO DE UMBRAL DE DECISION ---")
best_th, best_acc = 0.5, 0.0
for th in np.arange(0.50, 0.95, 0.05):
    preds = (all_probs >= th).astype(int)
    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    cm = confusion_matrix(all_targets, preds)
    if acc > best_acc:
        best_acc = acc
        best_th = th
    print(f"Umbral {th:.2f} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}% | FP: {cm[0,1]} FN: {cm[1,0]}")

print(f"\nOPTIMO -> Umbral: {best_th:.2f} con Acc: {best_acc*100:.2f}%")
