# 🧠 Proyecto de Redes Neuronales - Unidad IV  
## Optimización de Rutas Logísticas usando ADALINE y Perceptrón Multicapa (MLP)

Este proyecto implementa dos modelos de **redes neuronales artificiales supervisadas** aplicados a la optimización de rutas logísticas dentro de un sistema de transporte.

Los modelos desarrollados fueron:

- 📊 **Modelo ADALINE (Adaptive Linear Neuron)**  
- 🧠 **Perceptrón Multicapa (MLP)**  

Ambos modelos fueron desarrollados con fines académicos en la asignatura **Redes Neuronales**, permitiendo analizar patrones logísticos y seleccionar rutas óptimas basadas en variables operativas.

---

# 📌 Objetivo del proyecto

Desarrollar e implementar modelos de aprendizaje supervisado que permitan:

- Optimizar la selección de rutas logísticas  
- Clasificar rutas según condiciones operativas  
- Analizar patrones en variables logísticas  
- Comparar el desempeño entre modelos lineales y no lineales  
- Mejorar la toma de decisiones en sistemas de transporte  

---

# 🧩 Modelos implementados

## 🔷 Modelo 1 — ADALINE (Adaptive Linear Neuron)

El modelo ADALINE se utiliza como una red neuronal lineal que permite clasificar rutas logísticas a partir de relaciones matemáticas simples entre variables.

Este modelo funciona como una base de comparación para evaluar el rendimiento de modelos más avanzados.

### Características:

- **Tipo:** Red neuronal supervisada lineal  
- **Algoritmo:** Gradiente descendente estocástico (SGD)  
- **Función de pérdida:** Error cuadrático (`squared_error`)  
- **Normalización:** StandardScaler  
- **Dataset:** Archivo CSV con registros logísticos  
- **Precisión obtenida:** ≈ 71 %

### Dataset utilizado:

Archivo:
```text
dataset_rutas_logisticas.csv
```

Contiene registros simulados de rutas logísticas con múltiples variables operativas.

---

## 🔷 Modelo 2 — Perceptrón Multicapa (MLP)

El modelo Perceptrón Multicapa permite modelar relaciones complejas entre variables logísticas mediante el uso de múltiples capas de neuronas.

Este modelo mostró un desempeño superior en la clasificación de rutas óptimas.

### Características:

- **Tipo:** Red neuronal supervisada multicapa  
- **Capas ocultas:** 2  
- **Neuronas por capa:** 16 y 8  
- **Funciones de activación:** ReLU y Softmax  
- **Optimizador:** Adam  
- **Número de épocas:** 50  
- **Batch size:** 16  
- **Normalización:** StandardScaler  
- **Precisión obtenida:** ≈ 95 %

### Variables utilizadas:

Las variables de entrada representan condiciones logísticas:

- **V1 →** Distancia recorrida (km)  
- **V2 →** Nivel de tráfico  
- **V3 →** Tiempo estimado (minutos)  
- **V4 →** Consumo de combustible (litros)  

La variable de salida corresponde a:

- **Ruta óptima seleccionada**

---

# 📂 Estructura del proyecto
```text
proyecto_redes_neuronales/
├── ADALINE.py
├── MLP.py
├── dataset_rutas_logisticas_500.csv
└── README.md
```

---

# 📚 Librerías utilizadas

Este proyecto utiliza las siguientes librerías:

- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib` (opcional)

---

# 🖥️ Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado:

- Python 3.9 o superior  
- pip  

Instalar dependencias:

```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install tensorflow
pip install matplotlib
```

# 🚀 Instrucciones de Ejecución
# Modelo ADALINE
```bash
python ADALINE.py
```
El programa:

- Carga el dataset
- Normaliza los datos
- Entrena el modelo ADALINE
- Realiza predicciones
- Genera métricas de evaluación

# Modelo MLP
```bash
python MLP.py
```
El programa:

- Carga el dataset
- Normaliza los datos
- Entrena el modelo MLP
- Evalúa el desempeño
- Genera métricas de precisión
