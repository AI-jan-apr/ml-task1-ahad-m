# Machine Learning Task 1
## Breast Cancer — Binary Classification

---

## Objective

Build and compare multiple **binary classification** models to predict whether a tumor is:

- **0 — Malignant (Cancerous)**
- **1 — Benign (Non-cancerous)**

Models used:
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

> ⚠️ Feature scaling is NOT used in this task, as per the assignment requirements.

---

## Dataset

**Breast Cancer Wisconsin Dataset** — available directly in `scikit-learn`.

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 numerical |
| Target | Binary (0 = Malignant, 1 = Benign) |
| Missing values | None |

Features represent measurements extracted from digitized images of breast masses (radius, texture, area, smoothness, concavity, symmetry, etc.).

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
```

---

## Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Training: 455 samples | Testing: 114 samples
```

---

## Models Trained

### Logistic Regression
- Default (baseline)
- `max_iter=1000` — to allow full convergence

### SVM
- Default (baseline)
- `C=15` — to test the effect of a higher regularization margin

### KNN
- Default `k=5`
- `k=10` — to test a smoother decision boundary

---

## Evaluation Approach

> **Why `pos_label=0`?**  
> By default, scikit-learn treats class 1 as the positive class.  
> In this task, class **0 (Malignant)** is the critical class — missing a cancer case is the most dangerous error.  
> Therefore, all Precision / Recall / F1 scores are computed with `pos_label=0` to reflect **cancer detection performance**.

Confusion matrix layout used:

|  | Predicted Malignant (0) | Predicted Benign (1) |
|---|---|---|
| **Actual Malignant (0)** | TN | **FP ← Missed Cancer** |
| **Actual Benign (1)** | FN | TP |

---

## Results

| Model | Accuracy | Precision | **Recall** | F1 | Missed Cancers (FP) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (default) | 0.96 | 0.95 | 0.93 | 0.94 | 3 |
| **Logistic Regression (max_iter=1000)** | **0.97** | **0.98** | **0.95** | **0.96** | **2** |
| SVM (default) | 0.93 | 0.95 | 0.86 | 0.90 | 6 |
| SVM (C=15) | 0.94 | 0.95 | 0.88 | 0.91 | 5 |
| KNN (default, k=5) | 0.91 | 0.86 | 0.90 | 0.88 | 4 |
| KNN (k=10) | 0.93 | 0.90 | 0.90 | 0.90 | 4 |

---

## Conclusion

**Best model: Logistic Regression (max_iter=1000)**  
It achieved the highest Recall for Malignant (0.95) and the fewest missed cancers (FP = 2), making it the most suitable model for this medical task.

**Most important metric: Recall for Malignant (class 0)**  
In a medical context, a False Positive (malignant predicted as benign) means a cancer case goes undetected — which can delay diagnosis and put the patient's life at risk. Maximizing Recall minimizes this critical error.

---

## Project Structure

```
breast-cancer-binary-classification/
├── modeling.ipynb
└── README.md
```

---

## Requirements

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn