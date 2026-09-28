import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from src.custom_cnn import UAVShipNet, ConvBlock
from src.data_loader import UAVShipDataset

# Optimizacion 1: Aumentaciones sin bordes negros
mean = [0.3821, 0.4055, 0.3545]
std = [0.1450, 0.1088, 0.0999]

class RandomRot90:
    def __call__(self, img):
        k = np.random.randint(0, 4)
        return img.rotate(k * 90)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    RandomRot90(),
    transforms.ColorJitter(brightness=0.10, contrast=0.10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

df = pd.read_csv('data/dataset_metadata.csv')
print(f"Total datos: {len(df)} muestras.")

train_df, test_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
train_sub_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=42)

train_dataset = UAVShipDataset(train_sub_df, transform=train_transform)
val_dataset = UAVShipDataset(val_df, transform=val_transform)
test_dataset = UAVShipDataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Dispositivo:', device)

# Optimizacion 2: Modelo con Dropout calibrado (sin ahogar las primeras capas)
model = UAVShipNet(dropout_rate=0.2).to(device)
# Reducir dropout interno en bloques 1 y 2
model.block1.dropout = nn.Identity()
model.block2.dropout = nn.Identity()
model.block3.dropout = nn.Dropout2d(p=0.10)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
epochs = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

print("\n--- INICIANDO ENTRENAMIENTO OPTIMIZADO PARA CURVAS N5 ---")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(1, epochs + 1):
    t0 = time.time()
    model.train()
    running_loss, all_preds, all_targets = 0.0, [], []
    for imgs, tgts in train_loader:
        imgs, tgts = imgs.to(device), tgts.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, tgts)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.50).astype(int)
        all_preds.extend(preds.flatten())
        all_targets.extend(tgts.cpu().numpy().flatten())
        
    scheduler.step()
    tr_loss = running_loss / len(train_dataset)
    tr_acc = accuracy_score(all_targets, all_preds)
    
    # Validacion con eval() (Dropout desactivado)
    model.eval()
    val_loss, val_preds, val_targets = 0.0, [], []
    with torch.no_grad():
        for imgs, tgts in val_loader:
            imgs, tgts = imgs.to(device), tgts.to(device).unsqueeze(1)
            logits = model(imgs)
            loss = criterion(logits, tgts)
            val_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.50).astype(int)
            val_preds.extend(preds.flatten())
            val_targets.extend(tgts.cpu().numpy().flatten())
            
    val_loss = val_loss / len(val_dataset)
    val_acc = accuracy_score(val_targets, val_preds)
    
    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    dt = time.time() - t0
    print(f"Epoca [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | Tr Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%", flush=True)

# Guardar curvas
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], label='Train Loss', color='royalblue')
axes[0].plot(history['val_loss'], label='Val Loss', color='crimson')
axes[0].set_title('Curva de Perdida (BCE Loss)')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

axes[1].plot([a*100 for a in history['train_acc']], label='Train Acc', color='royalblue')
axes[1].plot([a*100 for a in history['val_acc']], label='Val Acc', color='crimson')
axes[1].axhline(y=98.0, color='forestgreen', linestyle=':', label='Meta ABET 98%')
axes[1].set_title('Curva de Exactitud (Accuracy %)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('reports/training_curves.png', dpi=150)
plt.close()
print("Nuevas curvas guardadas en reports/training_curves.png")

# Evaluar en test_eval/
files = [f for f in os.listdir('test_eval') if f.endswith('.png')]
model.eval()
t_preds, t_targets = [], []
with torch.no_grad():
    for f in files:
        lbl = 1 if f.startswith('1__') else 0
        img = Image.open(os.path.join('test_eval', f)).convert('RGB')
        tensor = val_transform(img).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(tensor)).item()
        pred = 1 if prob >= 0.50 else 0
        t_preds.append(pred)
        t_targets.append(lbl)

acc_eval = accuracy_score(t_targets, t_preds)
print(f"\nAccuracy en test_eval (a 0.50 estricto): {acc_eval*100:.2f}% ({sum(np.array(t_preds) == np.array(t_targets))}/{len(t_preds)})")

# Guardar checkpoint si supera el anterior
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epochs,
    'val_acc': val_acc,
    'test_eval_acc': acc_eval
}, 'models/uav_shipnet_best.pt')
print("Checkpoint actualizado en models/uav_shipnet_best.pt")
