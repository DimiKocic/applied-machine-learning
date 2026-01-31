# Applied Machine Learning — ITC6103 Final Project

This repository contains the final project for **ITC6103 – Applied Machine Learning (Winter 2025)**.

The project demonstrates a complete applied machine learning pipeline covering:

- Unsupervised learning (clustering and anomaly detection)
- Regression
- Multi-label classification
- Ensemble learning
- Explainable AI (SHAP)

A centralized command-line launcher (`_main.py`) is provided to run each task from a single entry point.

---

## Project Structure

AML/
│
├── _main.py # Central launcher
│
├── Clustering/

│ ├── K_Means.py

│ ├── DBSCAN.py

│ ├── kddcup.data_10_percent_corrected (download separately)

│ ├── x_tsne.npy

│ └── x_tsne_kmeans.npy
│
├── Regression/

│ ├── Regression.py

│ └── melb_data.csv (download separately)
│
├── NeuralNetwork/

│ ├── NeuralNetwork.py

│ ├── IMDB-F_CSV.csv (download separately)

│ ├── IMDB-F.arff (download separately)

│ ├── multi_output_mlp_model.pkl (generated locally)

│ └── multi_output_mlp_for_shap2.pkl (generated locally)
│
└── RandomForest/

├── RandomClassifier.py

├── IMDB-F_CSV.csv (download separately)

├── IMDB-F.arff (download separately)

├── clf.pkl (generated locally)

└── clf1.pkl (generated locally)


---

## Implemented Tasks

### 1. Anomaly Detection & Clustering (Unsupervised Learning)

Dataset: **KDDCup99 (10% corrected)**

Algorithms:
- K-Means
- DBSCAN

Techniques:
- One-hot encoding
- Feature scaling
- PCA dimensionality reduction
- t-SNE visualization
- Silhouette score
- Distance-based outlier detection

Outputs:
- Cluster plots
- Outlier detection
- Silhouette metrics

---

### 2. Regression (House Price Prediction)

Dataset:
- Melbourne Housing (`melb_data.csv`)

Models:
- Linear Regression
- Lasso Regression (with cross-validation)
- Polynomial Regression

Processing:
- Missing value imputation by region
- Rare-category grouping
- One-hot encoding
- Z-score outlier removal
- Feature standardization

Evaluation:
- MAE
- MSE
- R² score
- Cross-validation

Visualizations:
- Actual vs Predicted plots for all models

---

### 3. Multi-Label Classification + Explainable AI

Dataset:
- IMDB movie summaries (`IMDB-F_CSV.csv`)

Target genres:
- Drama
- Comedy
- Short
- Documentary
- Romance

Models:
- Multi-output MLP Neural Network
- Random Forest classifier

Pipeline:
- Dominant feature removal
- Standard scaling
- PCA (90% variance retained)
- Multi-label prediction
- ROC/AUC per genre
- Confusion matrices

Explainable AI:
- SHAP KernelExplainer
- Feature importance plots per genre

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib shap joblib scipy
```

How to Run

From the root directory:
```bash
python _main.py
```

You will be prompted to select:

1 - Clustering (DBSCAN)
2 - Clustering (K-Means)
3 - Regression
4 - Neural Network (MLP Multilabel)
5 - Random Forest


Each option executes the corresponding pipeline.

🔽 Datasets and Pretrained Models (Important)

- Due to size constraints, datasets and trained model files are not included in this repository.

- They must be downloaded or generated locally.

- KDDCup99 Dataset (Clustering)
  Download from:
```bash
https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data
```

- Required file:
  kddcup.data_10_percent_corrected


Place in:
```bash
Clustering/
```

IMDB Dataset (Multi-Label Classification)

Download from:
```bash
https://www.uco.es/kdis/mllresources/#FoodtruckDesc
```

Files:
IMDB-F_CSV.csv
IMDB-F.arff


Place both in:
```bash
NeuralNetwork/
RandomForest/
```

Melbourne Housing Dataset (Regression)
Download from:
```bash
https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market
```

File:
- melb_data.csv


Place in:
```bash
Regression/
```

🧠 Pretrained Models (.pkl)

Model files are not included:
```bash
multi_output_mlp_model.pkl
multi_output_mlp_for_shap2.pkl
clf.pkl
clf1.pkl
```

These must be generated locally.
Generate Neural Network Models

### Open NeuralNetwork/NeuralNetwork.py and uncomment the training sections, then run:
```bash
python NeuralNetwork/NeuralNetwork.py
```
-  Generate Random Forest Models

Run:
```bash
python RandomForest/RandomClassifier.py
```

After generation, _main.py will operate normally.

Outputs:
- Cluster visualizations
- Regression performance plots
- ROC curves
- Confusion matrices
- SHAP feature importance graphs
- All figures are automatically saved during execution.
