import os
import numpy as np
from PIL import Image

def analyze_folder(folder_path, max_samples=1000):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')][:max_samples]
    r_means, g_means, b_means = [], [], []
    r_stds, g_stds, b_stds = [], [], []
    contrasts = []
    
    for f in files:
        img = np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0
        r_means.append(img[:, :, 0].mean())
        g_means.append(img[:, :, 1].mean())
        b_means.append(img[:, :, 2].mean())
        r_stds.append(img[:, :, 0].std())
        g_stds.append(img[:, :, 1].std())
        b_stds.append(img[:, :, 2].std())
        contrasts.append(img.std())
        
    return {
        'count': len(files),
        'mean_rgb': (float(np.mean(r_means)), float(np.mean(g_means)), float(np.mean(b_means))),
        'std_rgb': (float(np.mean(r_stds)), float(np.mean(g_stds)), float(np.mean(b_stds))),
        'mean_contrast': float(np.mean(contrasts))
    }

if __name__ == '__main__':
    ships_stats = analyze_folder('data/raw/ships')
    noships_stats = analyze_folder('data/raw/no_ships', 1000)

    print('=== ESTADISTICAS EMPIRICAS DEL DATASET ===')
    print('BARCOS (Clase 1, n=1000):')
    print(f'  Media RGB: R={ships_stats["mean_rgb"][0]:.4f}, G={ships_stats["mean_rgb"][1]:.4f}, B={ships_stats["mean_rgb"][2]:.4f}')
    print(f'  Desv. Est. RGB: R={ships_stats["std_rgb"][0]:.4f}, G={ships_stats["std_rgb"][1]:.4f}, B={ships_stats["std_rgb"][2]:.4f}')
    print(f'  Contraste medio: {ships_stats["mean_contrast"]:.4f}')

    print('NO BARCOS (Clase 0, n=1000 muestra):')
    print(f'  Media RGB: R={noships_stats["mean_rgb"][0]:.4f}, G={noships_stats["mean_rgb"][1]:.4f}, B={noships_stats["mean_rgb"][2]:.4f}')
    print(f'  Desv. Est. RGB: R={noships_stats["std_rgb"][0]:.4f}, G={noships_stats["std_rgb"][1]:.4f}, B={noships_stats["std_rgb"][2]:.4f}')
    print(f'  Contraste medio: {noships_stats["mean_contrast"]:.4f}')
