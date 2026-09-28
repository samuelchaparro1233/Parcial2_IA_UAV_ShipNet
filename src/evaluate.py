"""
Modulo de evaluacion, inferencia en vivo y calculo de metricas en tiempo real
para pruebas ciegas (Requerimiento ABET E1, E3 y E4).
Soporta:
1. Carpeta local en disco (ruta fija o navegacion de archivos).
2. Carga interactiva mediante Browse / Drag & Drop (uploaded files).
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import time
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.custom_cnn import UAVShipNet

MODELS_DIR = 'models'
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'uav_shipnet_best.pt')

class ShipClassifierEvaluator:
    """
    Motor de inferencia en tiempo real para UAVShipNet (Custom CNN).
    """
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_model = None
        
        # Normalizacion espectral calibrada
        self.mean = [0.3821, 0.4055, 0.3545]
        self.std = [0.1450, 0.1088, 0.0999]
        self.transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.load_model()

    def load_model(self):
        if os.path.exists(CNN_MODEL_PATH):
            checkpoint = torch.load(CNN_MODEL_PATH, map_location=self.device)
            self.cnn_model = UAVShipNet(dropout_rate=0.0).to(self.device)
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.eval()
            print("UAVShipNet cargada exitosamente.", flush=True)
        else:
            raise FileNotFoundError(f"Modelo UAVShipNet no encontrado en {CNN_MODEL_PATH}")

    def predict_image(self, image_input, threshold=0.5):
        """
        Inferencia de una sola imagen (ruta de archivo, bytes o PIL Image).
        Retorna: (label_pred, prob_barco, latencia_ms, pil_image)
        """
        t0 = time.perf_counter()
        
        if isinstance(image_input, str):
            with Image.open(image_input) as img:
                pil_img = img.convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        else:
            # File-like object (ej. BytesIO de streamlit file_uploader)
            pil_img = Image.open(image_input).convert('RGB')
            
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn_model(tensor)
            prob = float(torch.sigmoid(logits).squeeze().cpu().numpy())
            
        pred_label = 1 if prob >= threshold else 0
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        return pred_label, prob, latency_ms, pil_img

    def predict_folder(self, folder_path, threshold=0.5):
        """
        Ejecuta inferencia sobre todas las imagenes de una carpeta.
        """
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_files = [
            f for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(valid_extensions)
        ]
        
        results = []
        total_latency = 0.0
        
        for f in image_files:
            img_path = os.path.join(folder_path, f)
            pred_label, prob, lat_ms, pil_img = self.predict_image(img_path, threshold=threshold)
            total_latency += lat_ms
            
            inferred_gt = None
            if f.startswith('1__'):
                inferred_gt = 1
            elif f.startswith('0__'):
                inferred_gt = 0
                
            results.append({
                'filename': f,
                'filepath': img_path,
                'pil_img': pil_img,
                'pred_label': pred_label,
                'pred_class': 'Barco' if pred_label == 1 else 'No Barco',
                'confidence': prob if pred_label == 1 else (1.0 - prob),
                'prob_ship': prob,
                'ground_truth': inferred_gt,
                'latency_ms': lat_ms
            })
            
        avg_latency = total_latency / len(results) if results else 0.0
        return results, avg_latency

    def predict_uploaded_files(self, uploaded_files, threshold=0.5):
        """
        Ejecuta inferencia sobre una lista de archivos subidos por Browse / Drag & Drop.
        """
        results = []
        total_latency = 0.0
        
        for uf in uploaded_files:
            fname = uf.name
            pred_label, prob, lat_ms, pil_img = self.predict_image(uf, threshold=threshold)
            total_latency += lat_ms
            
            inferred_gt = None
            if fname.startswith('1__'):
                inferred_gt = 1
            elif fname.startswith('0__'):
                inferred_gt = 0
                
            results.append({
                'filename': fname,
                'filepath': None,
                'pil_img': pil_img,
                'pred_label': pred_label,
                'pred_class': 'Barco' if pred_label == 1 else 'No Barco',
                'confidence': prob if pred_label == 1 else (1.0 - prob),
                'prob_ship': prob,
                'ground_truth': inferred_gt,
                'latency_ms': lat_ms
            })
            
        avg_latency = total_latency / len(results) if results else 0.0
        return results, avg_latency

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'confusion_matrix': cm.tolist()
        }
