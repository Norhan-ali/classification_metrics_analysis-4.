# Classification Metrics Analysis

## Project Overview
This project evaluates a Logistic Regression classification model using different classification metrics and visualization techniques. It focuses on understanding model performance, analyzing classification errors, and studying how changing the decision threshold affects Precision and Recall.

## Dataset
- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** Scikit-learn
- **Total Samples:** 569
- **Features:** 30 numerical features
- **Classes:**
  - 0: Malignant
  - 1: Benign

## Model and Evaluation Setup
A Logistic Regression model was trained using the same preprocessing approach as Notebook 1:
- 80% training data and 20% testing data.
- Stratified train-test split with `random_state=42`.
- Feature scaling using StandardScaler.
- Logistic Regression with `max_iter=1000`.

The test set contains 114 samples.

## Evaluation Results

| Metric | Score |
|---|---:|
| Accuracy | 98.25% |
| Precision | 98.61% |
| Recall | 98.61% |
| F1-Score | 98.61% |
| Specificity | 97.62% |
| ROC-AUC | 99.54% |

## Visualizations

### 1. Prediction Probability Distribution
A histogram showing the distribution of predicted probabilities for the Benign class.

### 2. Confusion Matrix
A heatmap displaying:
- True Negatives (TN): 41
- False Positives (FP): 1
- False Negatives (FN): 1
- True Positives (TP): 71

### 3. ROC Curve
The ROC curve evaluates the model's ability to distinguish between Malignant and Benign samples across different classification thresholds.

**ROC-AUC: 0.9954**

### 4. Threshold Comparison
The model was evaluated using three thresholds:

| Threshold | Precision | Recall | F1-Score |
|---:|---:|---:|---:|
| 0.3 | 0.9730 | 1.0000 | 0.9863 |
| 0.5 | 0.9861 | 0.9861 | 0.9861 |
| 0.7 | 0.9853 | 0.9306 | 0.9571 |

Lowering the threshold to 0.3 increased Recall to 100%, while increasing it to 0.7 reduced Recall to 93.06%.

## Key Findings
- The model achieved 98.25% accuracy on the test set.
- Manual metric calculations matched Scikit-learn results.
- The ROC-AUC score of 0.9954 indicates excellent class discrimination.
- Changing the classification threshold creates a trade-off between Precision and Recall.

## Repository Structure
```text
classification_metrics_analysis/
│
├── notebook_2.ipynb
└── README.md
```

## Tools and Libraries
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Conclusion
This project demonstrates how classification metrics provide a deeper understanding of model performance beyond accuracy. It also shows how adjusting the classification threshold affects the balance between Precision and Recall.
