"""
Script de integracion y aumento de dataset con FGSC-23 (GF-2 y Google Earth Satelital).
Estandariza a 80x80 px RGB y balancea el dataset a mas de 8,000 imagenes multi-sensor.
"""

import os
import io
import csv
import zipfile
import urllib.request
from PIL import Image

DATA_DIR = 'data'
RAW_SHIPS_DIR = os.path.join(DATA_DIR, 'raw', 'ships')
RAW_NOSHIPS_DIR = os.path.join(DATA_DIR, 'raw', 'no_ships')
METADATA_FILE = os.path.join(DATA_DIR, 'dataset_metadata.csv')

os.makedirs(RAW_SHIPS_DIR, exist_ok=True)
os.makedirs(RAW_NOSHIPS_DIR, exist_ok=True)

URL_FGSC23 = 'https://huggingface.co/datasets/jbourcier/fgsc23/resolve/main/FGSC-23.zip'

def download_and_integrate():
    print("Descargando dataset FGSC-23 (GF-2 y Google Earth) desde Hugging Face...")
    req = urllib.request.Request(URL_FGSC23, headers={'User-Agent': 'Mozilla/5.0'})
    zip_bytes = urllib.request.urlopen(req).read()
    print(f"Descarga completada ({len(zip_bytes)/(1024*1024):.2f} MB). Extrayendo y estandarizando a 80x80 RGB...")
    
    added_ships = 0
    added_noships = 0
    new_metadata_rows = []
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        image_names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for idx, name in enumerate(image_names):
            # En FGSC-23: subcarpeta '0' es non-ship, '1' a '22' son barcos
            is_non_ship = '/0/' in name
            label = 0 if is_non_ship else 1
            
            # Formato de nombre estandar con prefijo de clase
            base_fname = os.path.basename(name)
            prefix = "0__fgsc23_" if is_non_ship else "1__fgsc23_"
            standard_fname = f"{prefix}{base_fname.replace('.jpg', '.png')}"
            
            dest_dir = RAW_NOSHIPS_DIR if is_non_ship else RAW_SHIPS_DIR
            dest_path = os.path.join(dest_dir, standard_fname)
            
            # Leer imagen y redimensionar a 80x80 RGB
            raw_img_data = z.read(name)
            with Image.open(io.BytesIO(raw_img_data)) as img:
                img_rgb = img.convert('RGB')
                img_resized = img_rgb.resize((80, 80), Image.Resampling.LANCZOS)
                img_resized.save(dest_path, format='PNG')
                
            if is_non_ship:
                added_noships += 1
            else:
                added_ships += 1
                
            new_metadata_rows.append({
                'filename': standard_fname,
                'filepath': os.path.relpath(dest_path, start='.').replace('\\', '/'),
                'label': label,
                'class_name': 'no_ship' if is_non_ship else 'ship',
                'source': 'FGSC-23_GF2_GoogleEarth'
            })
            
            if (idx + 1) % 500 == 0 or (idx + 1) == len(image_names):
                print(f"  Procesadas {idx + 1}/{len(image_names)} imagenes...")
                
    print(f"\nIntegracion de FGSC-23 completada:")
    print(f"- Barcos nuevos agregados:     {added_ships}")
    print(f"- No-barcos nuevos agregados:  {added_noships}")
    print(f"- Total imagenes nuevas:       {added_ships + added_noships}")
    
    # Reconstruir dataset_metadata.csv con todo el catalogo unificado
    rebuild_metadata_catalog()

def rebuild_metadata_catalog():
    print("\nReconstruyendo catalogo unificado de metadatos (data/dataset_metadata.csv)...")
    all_rows = []
    
    for fname in sorted(os.listdir(RAW_SHIPS_DIR)):
        if fname.lower().endswith(('.png', '.jpg')):
            fpath = os.path.join(RAW_SHIPS_DIR, fname)
            source = 'FGSC-23_GF2' if 'fgsc23' in fname else 'PlanetScope'
            all_rows.append({
                'filename': fname,
                'filepath': os.path.relpath(fpath, start='.').replace('\\', '/'),
                'label': 1,
                'class_name': 'ship',
                'source': source
            })
            
    for fname in sorted(os.listdir(RAW_NOSHIPS_DIR)):
        if fname.lower().endswith(('.png', '.jpg')):
            fpath = os.path.join(RAW_NOSHIPS_DIR, fname)
            source = 'FGSC-23_GF2' if 'fgsc23' in fname else 'PlanetScope'
            all_rows.append({
                'filename': fname,
                'filepath': os.path.relpath(fpath, start='.').replace('\\', '/'),
                'label': 0,
                'class_name': 'no_ship',
                'source': source
            })
            
    with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'filepath', 'label', 'class_name', 'source'])
        writer.writeheader()
        writer.writerows(all_rows)
        
    ships_total = sum(1 for r in all_rows if r['label'] == 1)
    noships_total = sum(1 for r in all_rows if r['label'] == 0)
    print(f"Catalogo guardado con éxito:")
    print(f"- Total de imagenes en el dataset unificado: {len(all_rows)}")
    print(f"- Barcos (Clase 1):   {ships_total}")
    print(f"- No-Barcos (Clase 0): {noships_total}")

if __name__ == '__main__':
    download_and_integrate()
