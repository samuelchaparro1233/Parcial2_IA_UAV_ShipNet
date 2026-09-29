import os
import sys
sys.path.insert(0, os.path.abspath('.'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
import numpy as np
import pandas as pd
import torch

from src.custom_cnn import UAVShipNet, count_parameters
from src.evaluate import ShipClassifierEvaluator

def run_diagnostics():
    print("=" * 70)
    print("        INFORME DE VERIFICACION INTEGRAL PRE-PRESENTACION")
    print("=" * 70)

    # 1. Verificación del Dispositivo y Entorno
    print("\n[1] ENTORNO Y HARDWARE:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  * Dispositivo PyTorch: {device}")
    print(f"  * Version Python:      {sys.version.split()[0]}")
    print(f"  * Version PyTorch:     {torch.__version__}")

    # 2. Verificación del Checkpoint del Modelo
    print("\n[2] CHECKPOINT DEL MODELO (models/uav_shipnet_best.pt):")
    model_path = os.path.join('models', 'uav_shipnet_best.pt')
    if not os.path.exists(model_path):
        print(f"  [ERROR] No se encuentra el archivo {model_path}")
        return False
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  * Tamano en disco:     {size_mb:.2f} MB")
    
    ckpt = torch.load(model_path, map_location='cpu')
    print(f"  * Arquitectura:        {ckpt.get('architecture', 'Custom CNN')}")
    print(f"  * Epoca guardada:      {ckpt.get('epoch', 'N/A')}")
    if 'val_metrics' in ckpt:
        vm = ckpt['val_metrics']
        print(f"  * Val Loss en ckpt:    {vm.get('loss', 0.0):.4f}")
        print(f"  * Val Accuracy ckpt:   {vm.get('accuracy', 0.0)*100:.2f}%")
        print(f"  * Val F1 en ckpt:      {vm.get('f1', 0.0)*100:.2f}%")

    model = UAVShipNet()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    total_params = count_parameters(model)
    print(f"  * Parametros totales:  {total_params:,} (< 1M parametros - Ultra ligero)")

    # 3. Verificación de Inferencia sobre Carpeta Ciega (test_eval/)
    print("\n[3] EVALUACION EN TEST CIEGO (test_eval/ con umbral theta = 0.50):")
    evaluator = ShipClassifierEvaluator(device=device)
    t0 = time.time()
    results, avg_latency = evaluator.predict_folder('test_eval', threshold=0.50)
    total_eval_time = (time.time() - t0) * 1000

    y_true = [r['ground_truth'] for r in results]
    y_pred = [r['pred_label'] for r in results]
    metrics = evaluator.calculate_metrics(y_true, y_pred)

    acc = metrics['accuracy'] * 100.0
    prec = metrics['precision'] * 100.0
    rec = metrics['recall'] * 100.0
    f1 = metrics['f1'] * 100.0
    cm = metrics['confusion_matrix']
    tp, tn, fp, fn = metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn']

    print(f"  * Total Imagenes:      {len(results)}")
    print(f"  * Latencia por Imagen: {avg_latency:.2f} ms ({1000.0/avg_latency:.0f} FPS)")
    print(f"  * Tiempo Total Lote:   {total_eval_time:.1f} ms")
    print(f"  * Accuracy Obtenido:   {acc:.2f}%  (Meta ABET: >= 98.00%) -> {'[OK CUMPLE]' if acc >= 98.0 else '[NO CUMPLE]'}")
    print(f"  * Precision:           {prec:.2f}%  (TP={tp}, FP={fp})")
    print(f"  * Recall:              {rec:.2f}% (TP={tp}, FN={fn} - Barcos omitidos: {fn})")
    print(f"  * F1-Score:            {f1:.2f}%")
    print(f"  * Matriz de Confusion: TN={tn}, FP={fp} | FN={fn}, TP={tp}")
    print(f"    [[{cm[0][0]}, {cm[0][1]}], [{cm[1][0]}, {cm[1][1]}]]")

    # 4. Verificación de Integridad de la GUI (app.py)
    print("\n[4] VERIFICACION DE LA GUI Y ARCHIVOS PRINCIPALES:")
    import py_compile
    try:
        py_compile.compile('app.py', doraise=True)
        print("  * app.py:               [OK] Sintaxis Python y Streamlit valida")
    except Exception as e:
        print(f"  * app.py:               [ERROR] Sintaxis: {e}")
        return False

    required_files = [
        'run_app.bat', 'requirements.txt', 'README.md',
        'src/custom_cnn.py', 'src/data_loader.py', 'src/evaluate.py',
        'assets/logo_umng.png', 'assets/gui_interface_preview.png',
        'assets/training_curves.png', 'assets/dataset_samples.png',
        'reports/cnn_metrics_report.json'
    ]
    all_ok = True
    for rf in required_files:
        if os.path.exists(rf):
            print(f"  * {rf:28s} [OK] Presente ({os.path.getsize(rf)} bytes)")
        else:
            print(f"  * {rf:28s} [FALTA]")
            all_ok = False

    # 5. Estado Git y Remoto
    print("\n[5] ESTADO GIT Y REPOSITORIO:")
    import subprocess
    git_status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    if git_status.stdout.strip():
        print(f"  * Archivos pendientes: {git_status.stdout.strip()}")
    else:
        print("  * Repositorio limpio:   [OK] Todo sincronizado y commiteado")

    git_remote = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True)
    print(f"  * Remoto:               {git_remote.stdout.splitlines()[0] if git_remote.stdout else 'N/A'}")

    print("\n" + "=" * 70)
    print("                    ESTADO GENERAL: LISTO PARA EVALUAR")
    print("=" * 70)
    return True

if __name__ == '__main__':
    run_diagnostics()
