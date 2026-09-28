import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs('assets', exist_ok=True)

# Buscar imagenes reales representativas
ship_files = sorted(glob.glob('test_eval/1__*.png'))[:5]
noship_files = sorted(glob.glob('test_eval/0__*.png'))[:5]

fig, axes = plt.subplots(2, 5, figsize=(15, 6.8), facecolor='#060a14')

# Estilo y Titulo General
fig.suptitle(
    "CATÁLOGO DE MUESTRAS MULTI-SENSOR: BUQUES MERCANTES (CLASE 1) VS INFRAESTRUCTURA Y AGUA (CLASE 0)",
    color='#38bdf8', fontsize=12, fontweight='bold', y=0.98, fontfamily='sans-serif'
)

ship_subtitles = [
    "Carguero Portacontenedores\n(PlanetScope 3m GSD)",
    "Buque Mercante de Carga\n(Bahía de San Francisco)",
    "Granelero / Minerales\n(Dársena Comercial)",
    "Petrolero / Quimiquero\n(Tráfico Marítimo)",
    "Barcaza Mercante Fluvial\n(Canal de Acceso)"
]

noship_subtitles = [
    "Mar Abierto / Aguas Profundas\n(Fondo Marino Azul)",
    "Oleaje Costero y Espuma\n(Zona de Rompiente)",
    "Muelle Portuario de Concreto\n(Terminal de Carga)",
    "Costa Rocosa y Dique\n(Estructura Litoral)",
    "Aguas Portuarias Someras\n(Reflejos y Sedimentos)"
]

# Fila 1: Barcos
for idx, (f, sub) in enumerate(zip(ship_files, ship_subtitles)):
    ax = axes[0, idx]
    ax.set_facecolor('#0b1329')
    img = Image.open(f).convert('RGB')
    ax.imshow(img)
    ax.set_title(f"[CLASE 1: BARCO {idx+1}]\n{sub}", color='#34d399', fontsize=9, fontweight='bold', pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#10b981')
        spine.set_linewidth(2)

# Fila 2: No-Barcos
for idx, (f, sub) in enumerate(zip(noship_files, noship_subtitles)):
    ax = axes[1, idx]
    ax.set_facecolor('#0b1329')
    img = Image.open(f).convert('RGB')
    ax.imshow(img)
    ax.set_title(f"[CLASE 0: NO-BARCO {idx+1}]\n{sub}", color='#60a5fa', fontsize=9, fontweight='bold', pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#3b82f6')
        spine.set_linewidth(2)

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
output_path = 'assets/dataset_samples.png'
plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Figura de muestras generada exitosamente en {output_path} ({os.path.getsize(output_path)} bytes)")
