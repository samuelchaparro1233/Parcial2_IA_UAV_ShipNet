import os
import shutil
import glob
import random
import pandas as pd

random.seed(42)

# 1. Limpiar test_eval
os.makedirs('test_eval', exist_ok=True)
for f in os.listdir('test_eval'):
    file_path = os.path.join('test_eval', f)
    if os.path.isfile(file_path):
        os.remove(file_path)

# 2. Leer dataset_metadata.csv
df = pd.read_csv('data/dataset_metadata.csv')
print(f"Total muestras iniciales en catalogo: {len(df)}")

# Seleccionar 100 barcos y 100 no-barcos representativos (50% PlanetScope, 50% Rotterdam)
ships_planet = df[(df['label'] == 1) & (df['source'] == 'PlanetScope')].sample(50, random_state=42)
ships_rotterdam = df[(df['label'] == 1) & (df['source'] == 'FGSC23_Rotterdam_Cargo')].sample(50, random_state=42)
noships_planet = df[(df['label'] == 0) & (df['source'] == 'PlanetScope')].sample(50, random_state=42)
noships_port = df[(df['label'] == 0) & (df['source'] == 'FGSC23_Port_Infrastructure')].sample(50, random_state=42)

test_selection = pd.concat([ships_planet, ships_rotterdam, noships_planet, noships_port]).reset_index(drop=True)
print(f"Muestras seleccionadas para test_eval (100% ciegas): {len(test_selection)}")

# Copiar a test_eval
for _, row in test_selection.iterrows():
    src = row['filepath']
    dst = os.path.join('test_eval', row['filename'])
    shutil.copy2(src, dst)

test_eval_count = len(os.listdir('test_eval'))
print(f"Archivos copiados a test_eval/: {test_eval_count}")

# 3. Excluir estrictamente estas 200 imagenes de dataset_metadata.csv
test_filenames = set(test_selection['filename'])
train_df = df[~df['filename'].isin(test_filenames)].reset_index(drop=True)
print(f"Muestras restantes para entrenamiento/validacion de la red: {len(train_df)}")

train_df.to_csv('data/dataset_metadata.csv', index=False)

# Guardar registro oficial de la particion aislada para evidencia ABET
test_selection.to_csv('data/blind_test_isolated.csv', index=False)

# 4. Verificacion matematica de cero fuga de datos (Zero Data Leakage)
leakage = set(train_df['filename']).intersection(set(os.listdir('test_eval')))
print(f"VERIFICACION DE FUGA DE DATOS: {len(leakage)} archivos compartidos (DEBE SER 0).")
assert len(leakage) == 0, "Error crítico: Se detectó fuga de datos."
print("¡Garantía de aislamiento comprobada con éxito! test_eval/ es 100% ciega.")
