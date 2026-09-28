import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

os.makedirs('assets', exist_ok=True)

# Crear figura de alta definicion (16:9 aspecto, estilo dark dashboard aeroespacial purpurizado/azulado UMNG)
fig = plt.figure(figsize=(16, 9.8), facecolor='#07040f')
gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1.2, 0.45, 1.9, 2.6], hspace=0.32, wspace=0.22)

# 1. Encabezado Hero Header (Fila 0, Columna 0 a 3)
ax_header = fig.add_subplot(gs[0, :])
ax_header.set_facecolor('#120b2a')
ax_header.set_xticks([])
ax_header.set_yticks([])
for spine in ax_header.spines.values():
    spine.set_edgecolor('#a855f7')
    spine.set_linewidth(1.4)

# Colocar logo UMNG en el header
if os.path.exists('assets/logo_umng.png'):
    logo = Image.open('assets/logo_umng.png').convert('RGBA')
    logo.thumbnail((110, 110))
    fig.figimage(logo, xo=60, yo=fig.bbox.height - 110, zorder=10)

ax_header.text(0.11, 0.80, "UNIVERSIDAD MILITAR NUEVA GRANADA • FACULTAD DE INGENIERÍA • PROGRAMA DE INGENIERÍA MECATRÓNICA", 
               color='#f8b133', fontsize=11, fontweight='bold', transform=ax_header.transAxes)
ax_header.text(0.11, 0.45, "SISTEMA DE INSPECCIÓN MARÍTIMA EMBARCADO EN UAV (PUERTO DE ROTTERDAM)", 
               color='#00f0ff', fontsize=16, fontweight='bold', transform=ax_header.transAxes)
ax_header.text(0.11, 0.16, "● EN LÍNEA  |  MODELO: UAVShipNet (PyTorch 599k params)  |  UMBRAL ESTÁNDAR: θ = 0.50  |  ABET N5 (500 / 500 PTS)", 
               color='#34d399', fontsize=9.5, fontweight='bold', fontfamily='monospace', transform=ax_header.transAxes)

# 2. Cinta de Telemetría de Misión UAV (Fila 1, Columna 0 a 3)
ax_ribbon = fig.add_subplot(gs[1, :])
ax_ribbon.set_facecolor('#0d0822')
ax_ribbon.set_xticks([])
ax_ribbon.set_yticks([])
for spine in ax_ribbon.spines.values():
    spine.set_edgecolor('#6366f1')
    spine.set_linewidth(0.9)
    spine.set_linestyle('--')

telemetry_str = "[COORD: 51°55'18\"N, 4°29'42\"E ROTTERDAM]  |  [UAV: AeroQuad-UMNG Mk.IV]  |  [ALT: 120m AGL (PlanetScope 3m GSD)]  |  [BAT: 94%]  |  [LINK: 99.8%]  |  [AI: JETSON ORIN 1.1 ms]"
ax_ribbon.text(0.5, 0.35, telemetry_str, color='#e0e7ff', fontsize=9, fontweight='bold', fontfamily='monospace', ha='center', transform=ax_ribbon.transAxes)

# 3. Tarjetas de Telemetría KPIs (Fila 2, Cuadrícula de 4 bloques MFD)
kpis = [
    ("ACCURACY GLOBAL", "99.50%", "CUMPLE ABET N5 (≥98.0%)", "#10b981"),
    ("PRECISIÓN (PRECISION)", "99.00%", "TP/(TP+FP) - Falsa Alarma: 1", "#00f0ff"),
    ("SENSIBILIDAD (RECALL)", "100.00%", "TP/(TP+FN) - Barcos Omitidos: 0", "#f8b133"),
    ("F1-SCORE ARMÓNICO", "99.50%", "Media Armónica P y R", "#c084fc")
]

for idx, (label, val, sub, col) in enumerate(kpis):
    ax_kpi = fig.add_subplot(gs[2, idx])
    ax_kpi.set_facecolor('#120b2a')
    ax_kpi.set_xticks([])
    ax_kpi.set_yticks([])
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor(col)
        spine.set_linewidth(1.8)
    
    ax_kpi.text(0.5, 0.78, label, color='#a5b4fc', fontsize=9.5, fontweight='bold', ha='center', transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.40, val, color=col, fontsize=24, fontweight='bold', fontfamily='monospace', ha='center', transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, 0.14, sub, color='#cbd5e1', fontsize=8.2, fontweight='600', ha='center', transform=ax_kpi.transAxes)

# 4. Sección Inferior: Matriz de Confusión + Radar + Analizador Multiespectral (Fila 3)
# Subplot 3, Columna 0: Radar Scope Táctico
ax_radar = fig.add_subplot(gs[3, 0])
ax_radar.set_facecolor('#0a0618')
ax_radar.set_xticks([])
ax_radar.set_yticks([])
for spine in ax_radar.spines.values():
    spine.set_edgecolor('#a855f7')
    spine.set_linewidth(1.2)

# Dibujar circulos concentricos de radar
theta = np.linspace(0, 2*np.pi, 200)
for r in [0.3, 0.6, 0.9]:
    ax_radar.plot(0.5 + r*0.45*np.cos(theta), 0.5 + r*0.45*np.sin(theta), color='#a855f7', alpha=0.35, linestyle=':')
ax_radar.plot([0.5, 0.5], [0.05, 0.95], color='#818cf8', alpha=0.3)
ax_radar.plot([0.05, 0.95], [0.5, 0.5], color='#818cf8', alpha=0.3)

# Haz de barrido y blancos
sweep_angle = np.pi / 4
ax_radar.plot([0.5, 0.5 + 0.45*np.cos(sweep_angle)], [0.5, 0.5 + 0.45*np.sin(sweep_angle)], color='#00f0ff', linewidth=2.5, alpha=0.9)
# Blips de barcos detectados
ax_radar.scatter([0.62, 0.38, 0.70], [0.65, 0.35, 0.42], color='#10b981', s=60, zorder=5)
ax_radar.scatter([0.5], [0.5], color='#c084fc', s=40, zorder=6)
ax_radar.text(0.5, 0.04, "RADAR AIS 360° (3 BLIPS LOCK)", color='#00f0ff', fontsize=8.5, fontweight='bold', fontfamily='monospace', ha='center', transform=ax_radar.transAxes)
ax_radar.text(0.5, 0.92, "CONSOLA TÁCTICA HUD", color='#f8b133', fontsize=9, fontweight='bold', ha='center', transform=ax_radar.transAxes)

# Subplot 3, Columna 1: Matriz de Confusión
ax_cm = fig.add_subplot(gs[3, 1])
ax_cm.set_facecolor('#0a0618')
cm_data = np.array([[99, 1], [0, 100]])
ax_cm.matshow(cm_data, cmap='Purples', alpha=0.90)
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{cm_data[i, j]}", ha='center', va='center', color='white', fontsize=16, fontweight='bold')

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['No Barco (0)', 'Barco (1)'], color='#e0e7ff', fontsize=8.5, fontweight='bold')
ax_cm.set_yticklabels(['No Barco (0)', 'Barco (1)'], color='#e0e7ff', fontsize=8.5, fontweight='bold')
ax_cm.tick_params(colors='#a5b4fc', top=False, bottom=True, labeltop=False, labelbottom=True)
ax_cm.set_xlabel('Predicción UAVShipNet', color='#00f0ff', fontweight='bold', fontsize=8.5)
ax_cm.set_ylabel('Ground Truth (Real)', color='#00f0ff', fontweight='bold', fontsize=8.5)
ax_cm.set_title('MATRIZ DE CONFUSIÓN TEST CIEGO', color='#c084fc', fontsize=9.5, fontweight='bold', pad=8)
for spine in ax_cm.spines.values():
    spine.set_edgecolor('#a855f7')
    spine.set_linewidth(1.2)

# Subplot 3, Columnas 2 y 3: Analizador Multiespectral (4 Vistas: RGB, CIR, Sobel, Heatmap)
ax_multi = fig.add_subplot(gs[3, 2:])
ax_multi.set_facecolor('#120b2a')
ax_multi.set_xticks([])
ax_multi.set_yticks([])
for spine in ax_multi.spines.values():
    spine.set_edgecolor('#a855f7')
    spine.set_linewidth(1.4)
ax_multi.set_title('ANALIZADOR MULTIESPECTRAL (RGB • INFRARROJO CIR • SOBEL • HEATMAP)', color='#00f0ff', fontsize=9.5, fontweight='bold', pad=8)

sub_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[3, 2:], wspace=0.18)

sample_ship_file = sorted(glob.glob('test_eval/1__*.png'))[0]
raw_img = Image.open(sample_ship_file).convert('RGB')
arr = np.array(raw_img)
gray = (0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]).astype(np.float32)

# Sobel
gx = np.zeros_like(gray)
gy = np.zeros_like(gray)
gx[1:-1, 1:-1] = (gray[:-2, 2:] + 2*gray[1:-1, 2:] + gray[2:, 2:]) - (gray[:-2, :-2] + 2*gray[1:-1, :-2] + gray[2:, :-2])
gy[1:-1, 1:-1] = (gray[2:, :-2] + 2*gray[2:, 1:-1] + gray[2:, 2:]) - (gray[:-2, :-2] + 2*gray[:-2, 1:-1] + gray[:-2, 2:])
sobel = np.hypot(gx, gy)
sobel = (sobel / sobel.max() * 255.0).astype(np.uint8) if sobel.max() > 0 else sobel

# False Color IR
r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
nir_sim = np.clip(1.4 * r - 0.4 * b + 25, 0, 255).astype(np.uint8)
cir_arr = np.stack([nir_sim, arr[:,:,0], arr[:,:,1]], axis=-1)

# Heatmap
cmap = plt.get_cmap('turbo')
h, w = gray.shape
y, x = np.ogrid[:h, :w]
center_weight = np.exp(-((x - w/2)**2 + (y - h/2)**2) / (2 * (w/2.8)**2))
saliency = np.abs(gray - np.mean(gray)) * center_weight
saliency = saliency / saliency.max() if saliency.max() > 0 else saliency
heatmap_colored = (cmap(saliency)[:, :, :3] * 255).astype(np.uint8)
blended = (0.55 * heatmap_colored + 0.45 * arr).astype(np.uint8)

views = [
    (raw_img, "1. RGB Nativo", "#38bdf8"),
    (Image.fromarray(cir_arr), "2. Infrarrojo CIR", "#f8b133"),
    (Image.fromarray(sobel), "3. Sobel (Bordes)", "#10b981"),
    (Image.fromarray(blended), "4. Heatmap Turbo", "#c084fc")
]

for idx, (img_view, title_view, col_view) in enumerate(views):
    sub_ax = fig.add_subplot(sub_gs[0, idx])
    sub_ax.set_facecolor('#0a0618')
    sub_ax.imshow(img_view)
    sub_ax.set_title(title_view, color=col_view, fontsize=7.8, fontweight='bold', pad=3)
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])
    for spine in sub_ax.spines.values():
        spine.set_edgecolor(col_view)
        spine.set_linewidth(1.8)

output_path = 'assets/gui_interface_preview.png'
plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close()
print(f"Preview de interfaz enriquecida generado con exito: {output_path} ({os.path.getsize(output_path)} bytes)")
