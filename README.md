# Breast Cancer — Binary Classification

A machine learning project that predicts whether a tumor is **Malignant (Cancerous)** or **Benign (Non-cancerous)** using three classification models.

---

## Objective

Build and compare multiple binary classification models to predict tumor type:

| Label | Class |
|-------|-------|
| 0 | Malignant (Cancerous) |
| 1 | Benign (Non-cancerous) |

---

## Project Structure

```
breast-cancer-binary-classification/
├── modeling.ipynb
└── README.md
```

---

## Dataset

**Breast Cancer Wisconsin Dataset** available directly in scikit-learn.

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30 numerical |
| Target classes | 2 (Malignant, Benign) |
| Missing values | None |
| Malignant cases | 212 (37%) |
| Benign cases | 357 (63%) |

Features represent measurements extracted from digitized images of breast masses (e.g., radius, texture, area, smoothness, concavity, symmetry).

---

##  Models Used

- **Logistic Regression** — `max_iter=10000` to ensure convergence without feature scaling
- **Support Vector Machine (SVM)** — default parameters
- **K-Nearest Neighbors (KNN)** — default parameters (k=5)


---

##  Train-Test Split

| Parameter | Value |
|-----------|-------|
| test_size | 0.2 |
| random_state | 42 |
| stratify | y |

| Set | Samples |
|-----|---------|
| Training | 455 |
| Testing | 114 |

---

##  Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **0.9649** | 0.9595 | **0.9861** | **0.9726** |
| SVM | 0.9298 | 0.9211 | 0.9722 | 0.9459 |
| KNN | 0.9123 | 0.9429 | 0.9167 | 0.9296 |

---

##  Conclusion

### Best Performing Model
**Logistic Regression** achieved the best results across all metrics . highest accuracy (96.49%), highest recall (98.61%), and highest F1-Score (97.26%).

### Most Important Metric in Medical Context
**Recall** is the most critical metric in cancer detection. Missing a real cancerous case (False Negative) is far more dangerous than flagging a healthy patient for further tests (False Positive). A high Recall ensures that as few cancerous cases as possible go undetected, which is essential for early treatment and patient survival.

---

##  Requirements

```
scikit-learn
pandas
numpy
```

---

##  How to Run

1. Clone or download the project
2. Open `modeling.ipynb` in Jupyter Notebook
3. Run all cells in order
