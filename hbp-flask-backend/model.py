# ==============================================
# STEP 1: IMPORT LIBRARIES & DATA PREPROCESSING
# ==============================================
!pip install numpy pandas scikit-learn matplotlib krippendorff statsmodels
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
import krippendorff
from statsmodels.stats.inter_rater import fleiss_kappa

# Set working directory (modify if necessary)
os.chdir('C:\\Users\\dowus\\Dropbox\\Mr. Prince Otu\\_data analysis')

# Load and preprocess dataset
data = pd.read_csv("dataset_HBP00.csv").dropna()
#data.to_csv("dataset_HBP00_NAmissing.csv", index=False)
X, y = data.iloc[:, :-1], data.iloc[:, -1]  # Features & Labels

# Standardize Features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =======================================
# STEP 2: CLASS PRIOR ESTIMATION (Alpha)
# =======================================
clf_initial = LogisticRegression(solver='liblinear', random_state=0)
clf_initial.fit(X_train, y_train)
p_s_given_x = clf_initial.predict_proba(X_train)[:, 1]
alpha_hat = np.mean(p_s_given_x[y_train == 1])  
print(f"Estimated Class Prior (alpha): {alpha_hat:.4f}")

# ===================================
# STEP 3: DEFINE PU LEARNING MODELS
# ===================================
def train_pu_wlr():
    sample_weights = np.where(y_train == 1, 1, (1 - np.mean(y_train)) / np.mean(y_train))
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    probs = model.predict_proba(X_test)[:, 1] / alpha_hat
    probs = np.clip(probs, 0, 1)
    preds = (probs >= 0.5).astype(int)
    return model, probs, preds

def train_biased_svm():
    sample_weights_svm = np.where(y_train == 1, 1.0, 0.1)
    model = SVC(kernel="rbf", probability=True, random_state=0)
    model.fit(X_train, y_train, sample_weight=sample_weights_svm)
    probs = model.predict_proba(X_test)[:, 1] / alpha_hat
    probs = np.clip(probs, 0, 1)
    preds = (probs >= 0.5).astype(int)
    return model, probs, preds

def train_pu_bagging(n_estimators=10, neg_ratio=0.5):
    predictions_rf = np.zeros((len(y_test), n_estimators))
    for i in range(n_estimators):
        labeled_indices = np.where(y_train == 1)[0]
        unlabeled_indices = np.where(y_train == 0)[0]
        sampled_negatives = resample(unlabeled_indices, replace=False, n_samples=int(len(unlabeled_indices) * neg_ratio), random_state=i)
        sampled_indices = np.concatenate([labeled_indices, sampled_negatives])
        X_train_sampled = X_train[sampled_indices]
        y_train_sampled = y_train.to_numpy()[sampled_indices]
        rf_model = RandomForestClassifier(n_estimators=100, random_state=i)
        rf_model.fit(X_train_sampled, y_train_sampled)
        predictions_rf[:, i] = rf_model.predict_proba(X_test)[:, 1]
    probs = predictions_rf.mean(axis=1) / alpha_hat
    probs = np.clip(probs, 0, 1)
    preds = (probs >= 0.5).astype(int)
    return probs, preds

# Train models
_, y_probs_wlr, y_pred_wlr = train_pu_wlr()
_, y_probs_svm, y_pred_svm = train_biased_svm()
y_probs_rf, y_pred_rf = train_pu_bagging()

# ===================================
# STEP 4: HYPERPARAMETER TUNING
# ===================================
def tune_model(model, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="f1")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

best_wlr, best_params_wlr = tune_model(LogisticRegression(solver="liblinear", random_state=0), {"C": [0.01, 0.1, 1, 10, 100]})
best_svm, best_params_svm = tune_model(SVC(probability=True, random_state=0), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly"]})
best_rf, best_params_rf = tune_model(RandomForestClassifier(random_state=0), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]})

print(f"\n Best PU-WLR Parameters: {best_params_wlr}")
print(f" Best Biased SVM Parameters: {best_params_svm}")
print(f" Best PU Bagging Parameters: {best_params_rf}")

# =============================================
# STEP 5: MAKE PREDICTIONS WITH TUNED MODELS
# =============================================
def predict_tuned_model(model, X_test, alpha_hat):
    """Generate probability scores and binary predictions for a tuned model."""
    probs = model.predict_proba(X_test)[:, 1] / alpha_hat
    probs = np.clip(probs, 0, 1)
    preds = (probs >= 0.5).astype(int)
    return probs, preds

# Get predictions for tuned models
y_probs_wlr_tuned, y_pred_wlr_tuned = predict_tuned_model(best_wlr, X_test, alpha_hat)
y_probs_svm_tuned, y_pred_svm_tuned = predict_tuned_model(best_svm, X_test, alpha_hat)

# Retrain PU Bagging with best Random Forest parameters
def train_pu_bagging_tuned(best_rf, n_estimators=10, neg_ratio=0.5):
    predictions_rf = np.zeros((len(y_test), n_estimators))
    
    for i in range(n_estimators):
        labeled_indices = np.where(y_train == 1)[0]
        unlabeled_indices = np.where(y_train == 0)[0]
        sampled_negatives = resample(unlabeled_indices, replace=False, 
                                     n_samples=int(len(unlabeled_indices) * neg_ratio), 
                                     random_state=i)
        sampled_indices = np.concatenate([labeled_indices, sampled_negatives])
        X_train_sampled = X_train[sampled_indices]
        y_train_sampled = y_train.to_numpy()[sampled_indices]

        # Remove `random_state=i` since best_rf already contains a random state
        rf_model_tuned = RandomForestClassifier(**{k: v for k, v in best_rf.get_params().items() if k != 'random_state'})
        rf_model_tuned.fit(X_train_sampled, y_train_sampled)
        predictions_rf[:, i] = rf_model_tuned.predict_proba(X_test)[:, 1]

    probs = predictions_rf.mean(axis=1) / alpha_hat
    probs = np.clip(probs, 0, 1)
    preds = (probs >= 0.5).astype(int)
    return probs, preds


# Get predictions for tuned PU Bagging
y_probs_rf_tuned, y_pred_rf_tuned = train_pu_bagging_tuned(best_rf)


# =======================================
# STEP 6: MODEL PERFORMANCE EVALUATION
# =======================================
models = {
    "PU-WLR": (y_pred_wlr, y_probs_wlr),
    "Biased SVM": (y_pred_svm, y_probs_svm),
    "PU Bagging": (y_pred_rf, y_probs_rf)
}

for name, (y_pred, y_probs) in models.items():
    print(f"\n {name} Performance:\n" +
          f"Accuracy: {accuracy_score(y_test, y_pred):.4f} | "
          f"Precision: {precision_score(y_test, y_pred):.4f} | "
          f"Recall: {recall_score(y_test, y_pred):.4f} | "
          f"F1-score: {f1_score(y_test, y_pred):.4f} | "
          f"AUC: {roc_auc_score(y_test, y_probs):.4f}")

tuned_models = {
    "PU-WLR (Tuned)": (y_pred_wlr_tuned, y_probs_wlr_tuned),
    "Biased SVM (Tuned)": (y_pred_svm_tuned, y_probs_svm_tuned),
    "PU Bagging (Tuned)": (y_pred_rf_tuned, y_probs_rf_tuned)
}

for name, (y_pred, y_probs) in tuned_models.items():
    print(f"\n {name} Performance:\n" +
          f"Accuracy: {accuracy_score(y_test, y_pred):.4f} | "
          f"Precision: {precision_score(y_test, y_pred):.4f} | "
          f"Recall: {recall_score(y_test, y_pred):.4f} | "
          f"F1-score: {f1_score(y_test, y_pred):.4f} | "
          f"AUC: {roc_auc_score(y_test, y_probs):.4f}")

import joblib
# Save the trained Biased SVM model
joblib.dump(best_svm, "biased_svm_hbp.pkl")

# Save the scaler used for feature standardization
joblib.dump(scaler, "scaler.pkl")

print("✅ Biased SVM model and scaler saved successfully!")

