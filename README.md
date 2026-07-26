# Optimizadores de Regresión Logística: Descenso del Gradiente vs. Newton-Raphson

Este proyecto implementa y compara dos métodos numéricos de optimización sin restricciones (Descenso del Gradiente y Newton-Raphson) aplicados a la estimación de parámetros en un modelo de Regresión Logística para clasificación binaria. 

El objetivo es analizar experimentalmente el trade-off clásico de la optimización numérica: el costo computacional por iteración frente a la velocidad y estabilidad de convergencia.

---

## Características Principales

- Implementación desde Cero: Algoritmos desarrollados únicamente utilizando NumPy, sin depender de librerías de Machine Learning para la optimización.
- Arquitectura Modular (PyStatistics): Estructura orientada a objetos dentro de src/ que separa la lógica de los optimizadores (Optimizer, GradientDescent, NewtonRaphson) y del modelo (LogisticRegression).
- Validación Cruzada: Comparación de desempeño, estabilidad numérica y coeficientes frente al estándar de la industria (scikit-learn con solver L-BFGS).
- Métricas de Evaluación Integrales: Seguimiento de la función de costo (Log-Loss / Entropía Cruzada Binaria), Accuracy, Precisión, Recall, F1-Score y matrices de confusión.

---

## Estructura del Repositorio

Metodos-Numericos-Proyecto/
├── src/
│   ├── models/
│   │   └── logistic_regression.py   # Clase principal del modelo probabilístico
│   └── optimizers/
│       ├── optimizer.py             # Clase base / Interfaz abstracta
│       ├── gradient_descent.py      # Optimizador de 1.er orden
│       └── newton_raphson.py        # Optimizador de 2.º orden
├── notebooks/
│   └── benchmarking.ipynb          # Cuaderno interactivo con experimentos y gráficas
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación del proyecto

---

## Resumen de Resultados

Evaluado sobre un dataset sintético controlado de 200 observaciones (x1 = horas de estudio, x2 = puntaje previo):

| Método | Orden | Iteraciones | Tiempo (s) | Accuracy | Log-Loss Final |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Descenso del Gradiente | 1.º | 2000 (máx) | 0.03867 s | 0.78 | 1.1489 |
| Newton-Raphson | 2.º | 7 | 0.00067 s | 0.81 | 0.3925 |
| Scikit-Learn (L-BFGS) | Cuasi-Newton | 26 | 0.01479 s | 0.81 | 0.3925 |

### Principales Conclusiones
1. Newton-Raphson demostró ser claramente superior para baja dimensionalidad (3 parámetros), alcanzando convergencia cuadrática en solo 7 iteraciones e igualando exactamente los resultados de scikit-learn.
2. Descenso del Gradiente no logró converger de forma estable con un alpha = 0.01 debido a la diferencia de escala entre las variables de entrada, evidenciando la alta sensibilidad de los métodos de primer orden ante la falta de normalización de datos.

---

## Instalación y Ejecución

Sigue estos pasos para clonar el repositorio y reproducir los experimentos localmente:

### 1. Clonar el repositorio
git clone https://github.com/ngodoya/Metodos-Numericos-Proyecto.git
cd Metodos-Numericos-Proyecto

### 2. Crear y activar un entorno virtual
python -m venv venv

# En Linux/macOS:
source venv/bin/activate

# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1

### 3. Instalar dependencias
pip install -r requirements.txt

### 4. Ejecutar el Notebook de Pruebas
jupyter notebook

Abre el archivo notebooks/benchmarking.ipynb para ejecutar el benchmark completo y visualizar la generación de gráficos.

---

## Tecnologías Utilizadas

- Python 3.10+
- NumPy: Computación matricial y operaciones algebraicas vectorizadas.
- Pandas: Estructuración y consolidación tabular de métricas.
- Matplotlib & Seaborn: Visualización de fronteras de decisión y curvas de convergencia.
- Scikit-Learn: Usado exclusivamente como solución de referencia (baseline).

---


Universidad Nacional de Colombia — Facultad de Ingeniería (Sede Bogotá)