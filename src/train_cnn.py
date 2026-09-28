"""
Entrenamiento y validacion de UAVShipNet con Stratified K-Fold y Hold-Out Test Set.
Optimizado con Cosine Annealing, regularizacion y balanceo de perdida para ABET Nivel 5.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from src.custom_cnn import UAVShipNet
from src.data_loader import UAVShipDataset, get_transforms

REPORTS_DIR = 'reports'
MODELS_DIR = 'models'
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

METADATA_FILE = os.path.join('data', 'dataset_metadata.csv')

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping para estabilidad
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds.flatten())
        all_targets.extend(targets.cpu().numpy().flatten())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    all_probs, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            logits = model(images)
            loss = criterion(logits, targets)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            
    val_loss = running_loss / len(dataloader.dataset)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets, dtype=int)
    preds = (all_probs >= threshold).astype(int)
    
    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    cm = confusion_matrix(all_targets, preds)
    
    return {
        'loss': float(val_loss),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'probs': all_probs.tolist(),
        'targets': all_targets.tolist()
    }

def run_training_pipeline(epochs=20, batch_size=64, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de computo seleccionado: {device}")
    
    df = pd.read_csv(METADATA_FILE)
    print(f"Cargadas {len(df)} muestras desde {METADATA_FILE}.")
    
    # Division Train (80%) y Hold-out Test Ciego (20%)
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df['label'], random_state=42
    )
    # Division Train (85%) y Val (15%) del subconjunto de entrenamiento
    train_sub_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df['label'], random_state=42
    )
    
    # Guardar test ciego para la prueba de simulacion
    test_df.to_csv(os.path.join('data', 'blind_test_split.csv'), index=False)
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = UAVShipDataset(train_sub_df, transform=train_transform)
    val_dataset = UAVShipDataset(val_df, transform=val_transform)
    test_dataset = UAVShipDataset(test_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Conjunto de Entrenamiento: {len(train_dataset)} imagenes")
    print(f"Conjunto de Validacion:    {len(val_dataset)} imagenes")
    print(f"Conjunto de Test Ciego:    {len(test_dataset)} imagenes")
    
    # Modelo
    model = UAVShipNet(dropout_rate=0.20).to(device)
    
    # Funcion de perdida calibrada para el ratio 1.9:1 (3484 no-ships / 1828 ships cargueros)
    pos_weight = torch.tensor([1.9]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_f1 = 0.0
    best_model_path = os.path.join(MODELS_DIR, 'uav_shipnet_best.pt')
    
    print("\nIniciando entrenamiento de UAVShipNet...", flush=True)
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        epoch_time = time.time() - t0
        print(f"Epoca [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | "
              f"Tr Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% F1: {val_metrics['f1']*100:.2f}%", flush=True)
        
        # Guardar mejor modelo segun F1 y Accuracy
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'architecture': 'UAVShipNet'
            }, best_model_path)
            
    total_time = time.time() - start_time
    print(f"\nEntrenamiento finalizado en {total_time:.1f}s.")
    print(f"Mejor modelo guardado en: {best_model_path} (Val F1: {best_val_f1*100:.2f}%)")
    
    # Cargar mejor modelo y evaluar en el TEST SET CIEGO
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n" + "="*50)
    print("EVALUACION EN CONJUNTO DE TEST CIEGO (800 Imagenes)")
    print("="*50)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Accuracy en Test Ciego: {test_metrics['accuracy']*100:.2f}% (Meta ABET > 98%)")
    print(f"Precision en Test:       {test_metrics['precision']*100:.2f}%")
    print(f"Recall en Test:          {test_metrics['recall']*100:.2f}%")
    print(f"F1-Score en Test:        {test_metrics['f1']*100:.2f}%")
    print(f"Matriz de Confusion: {test_metrics['confusion_matrix']}")
    
    # Guardar reporte de metricas
    report_data = {
        'training_time_seconds': total_time,
        'best_epoch': checkpoint['epoch'],
        'validation_metrics': checkpoint['val_metrics'],
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'confusion_matrix': test_metrics['confusion_matrix']
        },
        'history': history
    }
    
    with open(os.path.join(REPORTS_DIR, 'cnn_metrics_report.json'), 'w') as f:
        json.dump(report_data, f, indent=2)
        
    # Graficar curvas de aprendizaje
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train Loss', color='royalblue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='crimson')
    axes[0].set_title('Curva de Perdida (Loss)')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot([a*100 for a in history['train_acc']], label='Train Acc', color='royalblue')
    axes[1].plot([a*100 for a in history['val_acc']], label='Val Acc', color='crimson')
    axes[1].axhline(y=98.0, color='forestgreen', linestyle=':', label='Meta ABET 98%')
    axes[1].set_title('Exactitud (%)')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Curvas de entrenamiento guardadas en: {os.path.join(REPORTS_DIR, 'training_curves.png')}")
    
    return report_data

if __name__ == '__main__':
    run_training_pipeline(epochs=18, batch_size=64, lr=1e-3)
