"""
Modulo de extraccion de descriptores visuales clasicos (HOG + Color Moments).
Sustenta la Linea Base (Baseline) requerida por la rubrica ABET (Criterio C1 / SO1).
"""

import numpy as np
from PIL import Image

def compute_hog(image_np, cell_size=8, n_bins=9):
    """
    Calcula el Histograma de Gradientes Orientados (HOG) para una imagen RGB o escala de grises.
    
    Parametros:
      image_np: ndarray (H, W, 3) o (H, W) en rango [0, 255] o [0, 1].
      cell_size: Tamano de celda espacial (por defecto 8x8 px).
      n_bins: Numero de particiones angulares [0, 180 deg) (por defecto 9).
      
    Retorna:
      vector 1D con los descriptores HOG normalizados en bloques 2x2.
    """
    if image_np.ndim == 3:
        # Conversion estandar a escala de grises segun luminancia ITU-R BT.601
        gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
    else:
        gray = image_np.astype(np.float32)
        
    gray = gray.astype(np.float32)
    H, W = gray.shape
    
    # Calculo de gradientes espaciales mediante diferencias centrales
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    
    # Magnitud y orientacion sin signo [0, 180)
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size
    bin_width = 180.0 / n_bins
    
    # Histograma por celda
    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            c_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            c_ang = ang[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            bin_idx = (c_ang / bin_width).astype(int) % n_bins
            for b in range(n_bins):
                cell_hist[i, j, b] = np.sum(c_mag[bin_idx == b])
                
    # Normalizacion L2-Hys en bloques deslizantes 2x2
    blocks = []
    for i in range(n_cells_y - 1):
        for j in range(n_cells_x - 1):
            block = cell_hist[i:i+2, j:j+2, :].ravel()
            norm = np.linalg.norm(block) + 1e-6
            block_norm = np.clip(block / norm, 0.0, 0.2)
            block_norm = block_norm / (np.linalg.norm(block_norm) + 1e-6)
            blocks.append(block_norm)
            
    return np.concatenate(blocks)

def compute_color_moments(image_np):
    """
    Calcula los primeros 3 momentos estadisticos de color por canal (RGB):
    1. Media (brillo promedio)
    2. Desviacion estandar (variabilidad / contraste)
    3. Asimetria (Skewness, inclinacion de la distribucion)
    """
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
        
    features = []
    for c in range(3):
        channel = img[:, :, c]
        mean = np.mean(channel)
        std = np.std(channel) + 1e-6
        skewness = np.mean(((channel - mean) / std) ** 3)
        features.extend([mean, std, skewness])
        
    return np.array(features, dtype=np.float32)

def extract_features_single(image_path_or_array):
    """
    Extrae el vector combinado de caracteristicas (HOG + Momentos de Color)
    para una imagen dada (ruta de archivo o ndarray RGB).
    """
    if isinstance(image_path_or_array, str):
        with Image.open(image_path_or_array) as img:
            arr = np.array(img.convert('RGB'))
    else:
        arr = np.array(image_path_or_array)
        
    hog_feat = compute_hog(arr, cell_size=8, n_bins=9)
    color_feat = compute_color_moments(arr)
    
    return np.hstack([hog_feat, color_feat])
