# 🚀 Credit Card Fraud Detection using Anomaly Detection, Clustering & Deep Learning

This repository presents an end-to-end **Credit Card Fraud Detection system** leveraging:

- 🔍 **Classical anomaly detection algorithms**  
- 🤖 **Deep learning Autoencoder**  
- 📊 **Clustering & density-based techniques**  
- 📉 **Dimensionality reduction (PCA, UMAP, t-SNE)**  
- 🧮 **Supervised baseline model for comparison**

The goal is to simulate **realistic fraud detection scenarios** where:
- Fraud cases are extremely rare (highly imbalanced problem)
- Labels may not be available (unsupervised focus)
- Robust anomaly scoring is essential

Dataset Source: **Kaggle — Credit Card Fraud Dataset**

---

# 📁 Repository Structure (Tabular Format)

Below is the **current live structure** based on your GitHub upload:

| Folder / File | Description |
|---------------|-------------|
| 📂 **model/** | Contains all trained machine learning models (`.pkl` files) |
| ├── dbscan_umap.pkl | DBSCAN clustering model on UMAP-reduced space |
| ├── elliptic_envelope.pkl | Robust covariance anomaly model |
| ├── isolation_forest.pkl | Isolation Forest anomaly detector |
| ├── lof.pkl | Local Outlier Factor (novelty mode) |
| ├── logistic_regression.pkl | Supervised baseline model |
| ├── oneclass_svm.pkl | One-Class SVM model |
| 📂 **notebook/** | Jupyter notebook containing full workflow |
| ├── credit-card-fraud-ml.ipynb | End-to-end implementation |
| 📂 **results/** | All performance outputs & visualizations |
| 📂 results/images | Visualization outputs |
| ├── PCA 2d visualization.png | PCA-based scatter plot |
| ├── UMAP 2d visualization.png | UMAP projection |
| ├── tsne 2d visualization.png | t-SNE embedding |
| ├── precision_recall_curves.png | PR comparison curves |
| 📂 results/metrics | Evaluation result tables |
| ├── anomaly_detection_results.csv | Raw anomaly scores |
| ├── final_metrics.csv | Combined model performance metrics |
| 📄 README.md | Project documentation |

---

# 🧠 Techniques Implemented

## 🔹 1. Dimensionality Reduction
-  PCA (2D & 10D)
-  UMAP
-  t-SNE

## 🔹 2. Clustering Algorithms
-  K-Means  
-  Gaussian Mixture Models (GMM)  
-  BIRCH  
-  DBSCAN (best with UMAP embeddings)  
-  Spectral Clustering  
-  Mean Shift  

Fraud density per cluster is used to compute **anomaly scores**.

## 🔹 3. Classical Anomaly Detection
| Model | Description |
|-------|-------------|
|  Isolation Forest | Random partitioning → isolates anomalies |
|  Local Outlier Factor | Density-based anomaly scoring |
|  One-Class SVM | Learns boundary of normal class |
|  Elliptic Envelope | Covariance-based anomaly detector |

## 🔹 4. Deep Learning Autoencoder
- Encoder → 16 → 8  
- Decoder → 16 → input  
- Trained only on **normal** transactions  
- Reconstruction error used as anomaly score  
- GPU-optimized training  

## 🔹 5. Supervised Baseline
- Logistic Regression with class-weight balancing  
- Used to compare supervised vs unsupervised performance  

---

# 📈 Results Summary

All performance metrics are stored in:

- 📄 `results/metrics/final_metrics.csv`
- 📄 `results/metrics/anomaly_detection_results.csv`

### 🔥 Key Insights:
- ⭐ **Isolation Forest** and ⭐ **Autoencoder** performed the best.
- 🟪 UMAP gave the clearest separation visually.
- ✔ Supervised Logistic Regression shows strong baseline performance.
- DBSCAN + UMAP performed meaningfully for cluster-based anomaly scoring.

---

# 📉 Visualizations Included

Stored in `results/images/`:

- 🎨 PCA 2D  
- 🎨 UMAP 2D  
- 🎨 t-SNE 2D  
- 📈 Precision–Recall curves  

These visualizations help explain:
- Data structure  
- Fraud distribution  
- Model discrimination capability  

---

# ⚙️ How to Run This Project

1️⃣ Clone repository:

```
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection-anomaly-models
```

2️⃣ Install dependencies:

```
pip install -r requirements.txt
```

3️⃣ Launch notebook:

```
jupyter notebook notebook/credit-card-fraud-ml.ipynb
```

4️⃣ Download dataset:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

5️⃣ Run all cells — outputs will appear automatically.

---

# 🎯 Why This Project Is Valuable to Recruiters

This project demonstrates your ability to:

- Work with **heavily imbalanced datasets**  
- Apply **unsupervised + deep learning** techniques  
- Build realistic **fraud detection pipelines**  
- Implement and evaluate **over 10 ML models**  
- Visualize high-dimensional data effectively  
- Organize ML projects professionally for GitHub  

This repository represents a complete, real-world-ready implementation of anomaly-based fraud detection.

---
