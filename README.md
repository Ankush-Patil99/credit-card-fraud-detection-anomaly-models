![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Models](https://img.shields.io/badge/ML-Models-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

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
# ✨ Project Highlights

- Implemented **12+ ML models** across anomaly detection, clustering, and deep learning  
- Built a GPU-optimized **Autoencoder** fraud detection system  
- Generated high-quality visualizations (PCA, UMAP, t-SNE, PR Curve)  
- Achieved strong **PR-AUC** with Isolation Forest & Autoencoder  
- Structured the project with clean folders + saved models + reproducible metrics  
- Used **unsupervised, semi-supervised, and supervised** learning approaches  
- Fully documented and GitHub-ready for recruiters  
---

# 📁 Repository Structure (Collapsible Format)

Below is the complete project structure with collapsible sections for easy navigation.

---

<details>
<summary><strong>📦 Models (Click to Expand)</strong></summary>

| File | Description |
|------|-------------|
| [dbscan_umap.pkl](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/model/dbscan_umap.pkl) | DBSCAN clustering model on UMAP-reduced space |
| [elliptic_envelope.pkl](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/model/elliptic_envelope.pkl) | Covariance-based anomaly detector |
| [isolation_forest.pkl](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/model/isolation_forest.pkl) | Isolation Forest anomaly model |
| [lof.pkl](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/model/lof.pkl) | Local Outlier Factor model |
| [logistic_regression.pkl](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/model/logistic_regression.pkl) | Supervised baseline model |
| [oneclass_svm.pkl](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/model/oneclass_svm.pkl) | One-Class SVM anomaly model |

</details>



<details>
<summary><strong>📒 Notebook (Click to Expand)</strong></summary>

| File | Description |
|------|-------------|
| [credit-card-fraud-ml.ipynb](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/notebook/credit-card-fraud-ml.ipynb) | Full end-to-end project notebook including preprocessing, DR, clustering, anomaly detection & autoencoder |

</details>


<details>
<summary><strong>📊 Results (Click to Expand)</strong></summary>

### 📁 Images
| Visualization | File |
|---------------|-------|
| **PCA 2D Plot** | [PCA 2D visualization](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/PCA%202d%20visualization.png) |
| **UMAP 2D Plot** | [UMAP 2D visualization](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/UMAP%202d%20visualization.png) |
| **Precision-Recall Curve** | [precision_recall_curves.png](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/precision_recall_curves.png) |
| **t-SNE 2D Plot** | [tsne 2d visualization](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/tsne%202d%20visualization.png) |



### 📁 Metrics
| File | Description |
|------|-------------|
| [anomaly_detection_results.csv](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/metrics/anomaly_detection_results.csv) | Raw anomaly scores for each model |
| [final_metrics.csv](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/metrics/final_metrics.csv) | Combined PR-AUC & ROC-AUC metrics |

</details>

---
# 🗺️ Project Workflow Overview

Data Loading
     →
Preprocessing
     →
Dimensionality Reduction
     →
Clustering
     →
Anomaly Detection
     →
Autoencoder
     →
Supervised Baseline
     →
Evaluation
     →
Saving Models & Results



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

<details>
<summary><strong>🎨 Click to Expand Visualizations</strong></summary>
<br>

All visual outputs are stored in:  
📁 `results/images/`

| Visualization | Description | Link |
|--------------|-------------|------|
| **PCA 2D Projection** | Linear dimensionality reduction showing coarse class separation | [View PCA Plot](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/PCA%202d%20visualization.png) |
| **UMAP 2D Projection** | Non-linear reduction capturing global + local structure (best separation) | [View UMAP Plot](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/UMAP%202d%20visualization.png) |
| **t-SNE 2D Projection** | High-detail embedding useful for cluster inspection | [View t-SNE Plot](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/tsne%202d%20visualization.png) |
| **Precision–Recall Curve** | Compares anomaly detectors on imbalanced fraud detection | [View PR Curve](https://github.com/Ankush-Patil99/credit-card-fraud-detection-anomaly-models/blob/main/credit-card-fraud-detection-anomaly-models/results/images/precision_recall_curves.png) |

### 🧠 Why These Visualizations Matter

- **PCA** → Provides quick linear separation insight  
- **UMAP** → Reveals true fraud clusters (best manifold learning method here)  
- **t-SNE** → Shows local anomaly behavior on sampled data  
- **PR Curve** → Demonstrates model performance on highly imbalanced fraud detection  

Together, these visualizations give a **holistic understanding** of:
- Data geometry  
- Fraud patterns  
- Model discrimination power  
- Where unsupervised methods excel or fail  

</details>


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
## 👤 Author
**Ankush Patil**  
Machine Learning & NLP Engineer  
📧 **Email**: ankpatil1203@gmail.com  
💼 **LinkedIn**: www.linkedin.com/in/ankush-patil-48989739a  
🌐 **GitHub**: https://github.com/Ankush-Patil99  
---
