# 🏥 Pulmonary Hypertension Detection with Gradient Boosting (CatBoost)

**A machine learning model** that analyzes ECG signals to detect pulmonary hypertension (PH) using **gradient boosting (CatBoost)**. The project includes data preprocessing, model training, and a user-friendly GUI for predictions.

---

## 🌟 Features
- **Automated file sorting** (`SDLA30` for non-PH, `SDLA50` for PH cases)
- **ECG signal processing** with R-peak detection and cycle interpolation
- **CatBoost classifier** with optimized hyperparameters
- **Interactive GUI** for real-time predictions
- **Detailed reports** with probability distributions

```python
# Model Architecture
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    verbose=100
)
