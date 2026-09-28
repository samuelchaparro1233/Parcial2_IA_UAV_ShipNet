<div align="center">
  <img src="assets/logo_umng.png" width="130" alt="Escudo Universidad Militar Nueva Granada">
  <h1>UNIVERSIDAD MILITAR NUEVA GRANADA</h1>
  <h3>Facultad de Ingeniería • Programa de Ingeniería Mecatrónica</h3>
  <h4>Inteligencia Artificial — Proyecto 2 (Segundo Corte)</h4>
  <p><strong>Evaluación según Rúbrica ABET (Student Outcomes SO1 y SO6)</strong></p>
  <p><strong>Meta de Desempeño:</strong> Nivel N5 (Excelente / 500 puntos / Calificación 5.0) — <em>Accuracy > 98.0% en test desconocido</em></p>
</div>

---

## 🛰️ Sistema de Inspección Marítima Embarcado en UAV (Drones)
### Clasificador Binario de Embarcaciones en Imágenes Satelitales (80×80 px, RGB)
**Modelo Propio:** `UAVShipNet` (Red Neuronal Convolucional implementada desde cero en PyTorch)  
**Entorno Operativo:** Monitoreo Autónomo de la Flota Mercante del Puerto de Rotterdam  

---

## 1. Contexto Mecatrónico y Justificación Operacional

### A. Problemática Operacional
Los sistemas tradicionales de gestión de tráfico marítimo (VTS basado en radar costero y el Sistema de Identificación Automática AIS) sufren de dos limitaciones críticas:
1. **Latencia del AIS (2 a 180 s)** e invisibilidad de embarcaciones menores, botes sin transpondedor o buques no cooperativos (*dark vessels*).
2. **Ruido de eco parásito (*clutter*) del radar** en proximidades de muelles, espigones y dársenas portuarias.

### B. Solución Embarcada en Dron
Integración de un clasificador de visión artificial de alta velocidad directamente en el computador de abordo del dron (*Edge AI* / NVIDIA Jetson), permitiendo verificar ópticamente la presencia de buques en zonas portuarias con una latencia inferior a 1.5 ms y sin saturar el canal de radiofrecuencia a tierra.

* **Referencias Técnicas y Académicas:**
  * Dahana, U. & Gurning, R.O.S. (2020). *Maritime Aerial Surveillance: Integration Manual Identification System to Automatic Identification System*, IOP Conf. Ser.: Earth Environ. Sci. 557 012014.
  * Port of Rotterdam Drone Surveillance Initiative (*The Maritime Executive*).
  * Zhang, X. et al. (2020). *FGSC-23: A new benchmark for fine-grained ship classification in optical remote sensing images*, IEEE JSTARS.

---

## 2. Estrategia de Datos y Aislamiento de Test Ciego (Zero Leakage)

Para garantizar la máxima integridad científica y evitar cualquier fuga de datos (*Data Leakage*), el conjunto de evaluación está estrictamente aislado del entrenamiento:

| Subconjunto | Sensor / Plataforma | Composición Semántica | Cantidad | Estado de Aislamiento |
| :--- | :--- | :--- | :---: | :---: |
| **Entrenamiento / Validación** | PlanetScope (3m GSD) + Flota Mercante Rotterdam | 1,728 Cargueros + 3,384 Dársenas/Muelles | **5,112** | Empleadas para ajustar los pesos del modelo |
| **Test Ciego Desconocido (`test_eval/`)** | PlanetScope (3m GSD Nativo) | 100 Buques Mercantes + 100 Fondos Marinos | **200** | **100% Aisladas (Jamás vistas por la red)** |
| **TOTAL GENERAL** | **Multi-Sensor Global** | **Estandarizado 80×80 px RGB** | **5,312** | **Cero fuga de datos garantizada** |

* **Selección Temática Rotterdam:** Se priorizaron barcos comerciales (portacontenedores, graneleros, petroleros, metaneros GNL y barcazas del Rin), excluyendo por completo embarcaciones de guerra (portaaviones, destructores, fragatas y submarinos).

---

## 3. Arquitectura Propia `UAVShipNet` (PyTorch)

Diseñada desde cero bajo restricciones de bajo peso y alta velocidad para computadores de abordo:

* **Entrada:** Tensor $(B, 3, 80, 80)$ normalizado espectralmente ($\mu = [0.382, 0.406, 0.355]$, $\sigma = [0.145, 0.109, 0.100]$).
* **Bloques Convolucionales:** 4 etapas jerárquicas $(32 \rightarrow 64 \rightarrow 128 \rightarrow 256)$ con doble convolución $3\times 3$, Batch Normalization y activaciones LeakyReLU($\alpha=0.1$).
* **Preservación Temprana:** Eliminación de dropout en Bloques 1 y 2 para preservar bordes y contornos del casco; regularización suave ($p=0.10$) en Bloque 3 y en la cabeza lineal ($p=0.20$).
* **Global Average Pooling (GAP):** Reduce la dimensión espacial a $(B, 256, 1, 1)$, otorgando total invariancia a traslaciones espaciales en la cámara del UAV.
* **Presupuesto Computacional:** Solo **599,521 parámetros entrenables** (~2.4 MB en disco). Permite inferencias a **>250 FPS** (< 1.5 ms por imagen).

---

## 4. Desempeño y Cumplimiento de Criterios ABET (Nivel N5)

| Criterio ABET | Peso | Indicador de Desempeño | Evidencias en el Sistema |
| :--- | :---: | :--- | :---: |
| **C1. Metodología y Técnicas de Optimización ML** | 50% | **SO1 / RAE-140:** Uso de técnicas de ML para optimizar el desempeño mecatrónico. | **E1 y E2:** UI interactiva, optimizador AdamW, Cosine Annealing, Data Augmentation ortogonal (0°, 90°, 180°, 270°). |
| **C2. Evaluación, Validación Cruzada e Inferencia en Vivo** | 50% | **SO6 / RAE-144:** Inferencias sobre el desempeño con pruebas y métricas cuantitativas. | **E3 y E4:** Inferencia en vivo sobre carpeta ciega, matriz de confusión, accuracy en tiempo real y latencia sub-milimétrica. |

### Resultados en la Carpeta Ciega (`test_eval/` con $\theta = 0.50$ Estándar):

| Métrica de Desempeño | Resultado Obtenido | Requisito Rúbrica ABET N5 | Estado |
| :--- | :---: | :---: | :---: |
| **Accuracy en Test Ciego** | **99.50%** (199 / 200 correctos) | $\ge 98.00\%$ | **CUMPLIDO CON EXCELENCIA (5.0)** |
| **Sensibilidad (Recall)** | **100.00%** (100 / 100 barcos) | $\ge 95.00\%$ | Cero barcos perdidos |
| **Precisión** | **99.00%** (solo 1 falsa alarma) | $\ge 95.00\%$ | Mínima tasa de falsos positivos |
| **F1-Score** | **99.50%** | $\ge 95.00\%$ | Balance armónico óptimo |
| **Latencia de Inferencia** | **~1.1 ms / imagen** | $< 15\text{ ms}$ | Apto para control en tiempo real |

$$\text{Matriz de Confusión en Test Ciego:} \quad \begin{bmatrix} 99 & 1 \\ 0 & 100 \end{bmatrix}$$

---

## 5. Instrucciones de Instalación y Ejecución

### Opción Rápida (Windows):
Simplemente haz doble clic en el archivo ejecutable por lotes:
```cmd
run_app.bat
```

### Opción Manual desde Terminal:
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/samuelchaparro1233/Parcial2_IA_UAV_ShipNet.git
   cd Parcial2_IA_UAV_ShipNet
   ```
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Iniciar la interfaz gráfica:
   ```bash
   streamlit run app.py
   ```
4. Abrir en el navegador: `http://localhost:8501`.

---

## 6. Estructura del Repositorio

```
├── app.py                      # Interfaz gráfica interactiva Streamlit (E1 a E4)
├── run_app.bat                 # Lanzador automático en Windows
├── requirements.txt            # Dependencias del proyecto
├── README.md                   # Documentación técnica completa (ABET N5)
├── .gitignore                  # Reglas de exclusión de archivos masivos
│
├── assets/
│   ├── logo_umng.png           # Escudo oficial de la UMNG (alta resolución)
│   └── logo_umng.svg           # Escudo vectorial UMNG
│
├── models/
│   └── uav_shipnet_best.pt     # Checkpoint entrenado de UAVShipNet (PyTorch)
│
├── reports/
│   ├── training_curves.png     # Curvas de aprendizaje (BCE Loss y Accuracy)
│   └── cnn_metrics_report.json # Reporte cuantitativo de métricas
│
├── src/
│   ├── custom_cnn.py           # Arquitectura de la red neuronal UAVShipNet
│   ├── data_loader.py          # Data augmentation y precarga de imágenes
│   └── evaluate.py             # Motor de inferencia en tiempo real
│
├── test_eval/                  # 200 imágenes aisladas para la prueba del docente (100% ciegas)
└── data/
    └── dataset_metadata.csv    # Catálogo de metadatos de las 5,112 imágenes de entrenamiento
```

---

## 7. Referencias Bibliográficas, Datasets y Atribución (Open Science)

Este proyecto se adhiere a los principios de **ética académica, reproducibilidad científica y licenciamiento de datos abiertos**:

### A. Datasets Empleados

1. **Ships in Satellite Imagery (Dataset Core)**
   * **Autor / Creador:** Robert Hammell (`rhammell`) & Planet Labs Inc.
   * **Plataforma:** Kaggle Datasets (2018).
   * **Enlace Oficial:** [https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)
   * **Licencia:** Open Database License (ODbL) / CC BY-SA 4.0.
   * **Descripción:** 4,000 imágenes RGB de 80×80 px extraídas de la constelación PlanetScope (Dove cubesats a 3 m GSD) sobre la Bahía de San Francisco y el Puerto de Oakland.
   * **Cita Académica:**
     ```bibtex
     @misc{hammell2018ships,
       title={Ships in Satellite Imagery: 80x80 RGB images of ships and non-ships},
       author={Hammell, Robert and Planet Labs},
       year={2018},
       publisher={Kaggle},
       url={https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery}
     }
     ```

2. **FGSC-23: Fine-Grained Ship Classification in Optical Remote Sensing Images**
   * **Autores:** Xiaoqiang Zhang, Xiangxuan Ge, et al. (School of Computer Science and Engineering, Beihang University).
   * **Sensor:** Satélite Gaofen-2 (GF-2) y Google Earth (0.8 m GSD).
   * **Repositorio Hugging Face:** [https://huggingface.co/datasets/jbourcier/fgsc23](https://huggingface.co/datasets/jbourcier/fgsc23)
   * **Repositorio GitHub:** [https://github.com/dgy82/Satellite-Imagery-Datasets-Containing-Ships](https://github.com/dgy82/Satellite-Imagery-Datasets-Containing-Ships)
   * **Uso en el Proyecto:** Se filtraron exclusivamente los buques comerciales de la marina mercante (portacontenedores, graneleros, petroleros, metaneros y barcazas fluviales) y fondos portuarios para contextualizar el Puerto de Rotterdam.
   * **Cita Académica:**
     ```bibtex
     @article{zhang2020fgsc23,
       title={A New Benchmark and an Attribute-Guided Multilevel Feature Representation Network for Fine-Grained Ship Classification in Optical Remote Sensing Images},
       author={Zhang, Xiaoqiang and Ge, Xiangxuan and others},
       journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
       volume={13},
       pages={2970--2985},
       year={2020},
       publisher={IEEE},
       doi={10.1109/JSTARS.2020.2996950}
     }
     ```

3. **Ship Detection using Faster R-CNN: Part 1**
   * **Autor:** Aditya Jain (`adityajn105`).
   * **Plataforma:** Kaggle Code (2020).
   * **Enlace:** [https://www.kaggle.com/code/adityajn105/ship-detection-using-faster-r-cnn-part-1](https://www.kaggle.com/code/adityajn105/ship-detection-using-faster-r-cnn-part-1)
   * **Aporte Metodológico:** Estudio de propuestas de región (ROIs) y tratamiento del desbalance de clases (1:3) en visión por computador marítima.

### B. Literatura Operacional y Mecatrónica

* **Dahana, U., & Gurning, R. O. S. (2020).** *Maritime Aerial Surveillance: Integration Manual Identification System to Automatic Identification System*. IOP Conference Series: Earth and Environmental Science, 557(1), 012014. DOI: [10.1088/1755-1315/557/1/012014](https://doi.org/10.1088/1755-1315/557/1/012014).
* **Port of Rotterdam Authority (2022).** *Drone-based Smart Port Surveillance: Autonomous Inspection Operations in Deep-sea Terminals*. The Maritime Executive.
* **Gallego, A.-J., Pertusa, A., & Gil, P. (2018).** *Automatic Ship Classification from Optical Aerial Images with Convolutional Neural Networks*. Remote Sensing, 10(4), 511. DOI: [10.3390/rs10040511](https://doi.org/10.3390/rs10040511).

### C. Identidad Institucional
* **Universidad Militar Nueva Granada (UMNG):** Portal institucional oficial: [https://www.umng.edu.co](https://www.umng.edu.co).
* **Escudo Oficial:** Wikimedia Commons, [Archivo: Escudo oficial Universidad Militar Nueva Granada](https://commons.wikimedia.org/wiki/File:Escudo_oficial_Universidad_Militar_Nueva_Granada.svg), bajo licencia de identidad corporativa académica.

---
*Desarrollado para la Facultad de Ingeniería de la Universidad Militar Nueva Granada (UMNG).*