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


# ==================================================
# STEP 7: INTER-MODEL AGREEMENT RATE BETWEEN MODELS
# ==================================================
def compute_agreement(hidden_probs, threshold=0.55):
    hidden_labels = [(probs >= threshold).astype(int) for probs in hidden_probs]
    hidden_matrix = np.vstack(hidden_labels).T
    agreement_rate = np.mean((hidden_matrix[:, 0] == hidden_matrix[:, 1]) & (hidden_matrix[:, 0] == hidden_matrix[:, 2])) * 100
    kripp_alpha = krippendorff.alpha(hidden_matrix.T)
    fleiss_k = fleiss_kappa(np.array([[np.sum(row == 0), np.sum(row == 1)] for row in hidden_matrix]))
    return agreement_rate, kripp_alpha, fleiss_k

agreement_rate, kripp_alpha, fleiss_k = compute_agreement([y_probs_wlr, y_probs_svm, y_probs_rf])

def compute_agreement(hidden_probs, threshold=0.55):
    hidden_labels = [(probs >= threshold).astype(int) for probs in hidden_probs]
    hidden_matrix = np.vstack(hidden_labels).T
    agreement_rate = np.mean((hidden_matrix[:, 0] == hidden_matrix[:, 1]) & (hidden_matrix[:, 0] == hidden_matrix[:, 2])) * 100
    kripp_alpha = krippendorff.alpha(hidden_matrix.T)
    fleiss_k = fleiss_kappa(np.array([[np.sum(row == 0), np.sum(row == 1)] for row in hidden_matrix]))
    return agreement_rate, kripp_alpha, fleiss_k

agreement_rate_tuned, kripp_alpha_tuned, fleiss_k_tuned = compute_agreement([y_probs_wlr_tuned, y_probs_svm_tuned, y_probs_rf_tuned])

print(f"\nAgreement Rate Between PU Models: {agreement_rate:.2f}%")
print(f"Krippendorff’s Alpha: {kripp_alpha:.4f}")
print(f"Fleiss' Kappa: {fleiss_k:.4f}")

print(f"\n Agreement Rate Between Tuned PU Models: {agreement_rate_tuned:.2f}%")
print(f"Krippendorff’s Alpha (Tuned): {kripp_alpha_tuned:.4f}")
print(f"Fleiss' Kappa (Tuned): {fleiss_k_tuned:.4f}")



# ===========================================================
# STEP 8: FINAL PLOTS (ROC + PRECISION-RECALL)
# ===========================================================

plt.rcParams["axes.edgecolor"] = "black"  # Black border for axes
plt.rcParams["axes.linewidth"] = 0.9  # Thin border lines

# Define Custom Hex Color Palettes
base_colors = {
    "PU-WLR": "#083864",  # Dark Blue
    "Biased SVM": "#169bbd",  # Cyan
    "PU Bagging": "#1c8065"  # Green
}
tuned_colors = {
    "PU-WLR (Tuned)": "#083864",  # Dark Blue
    "Biased SVM (Tuned)": "#169bbd",  # Cyan
    "PU Bagging (Tuned)": "#1c8065"  # Green
}


font_size = 12  # Text size
legend_size = 12  # Legend text size
axis_tick_interval = 0.2  # Axis tick mark interval

# =====================================
# FIGURE 1: BASE MODELS (A & B)
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (A) ROC Curve (Base Models)
ax = axes[0]
for model, probs in zip(base_colors.keys(), [y_probs_wlr, y_probs_svm, y_probs_rf]):
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=f"{model} (AUC: {roc_auc_score(y_test, probs):.3f})", 
            color=base_colors[model], linewidth=1.5)

ax.plot([0, 1], [0, 1], linestyle="--", color="red", linewidth=1)  # Diagonal reference line
ax.set_xlabel("False Positive Rate", fontsize=font_size)
ax.set_ylabel("True Positive Rate", fontsize=font_size)
ax.legend(fontsize=legend_size, frameon=True)
ax.set_xticks(np.arange(0, 1.1, axis_tick_interval))
ax.set_yticks(np.arange(0, 1.1, axis_tick_interval))
ax.tick_params(axis="both", which="major", labelsize=font_size - 1)

# (B) Precision-Recall Curve (Base Models)
ax = axes[1]
for model, probs in zip(base_colors.keys(), [y_probs_wlr, y_probs_svm, y_probs_rf]):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ax.plot(recall, precision, label=model, 
            color=base_colors[model], linewidth=1.5)

ax.plot([0, 1], [0.3, 0.3], linestyle="--", color="red", linewidth=1)
ax.set_xlabel("Recall", fontsize=font_size)
ax.set_ylabel("Precision", fontsize=font_size)
ax.legend(fontsize=legend_size, frameon=True, loc="lower right")
ax.set_xticks(np.arange(0, 1.1, axis_tick_interval))
ax.set_yticks(np.arange(0, 1.1, axis_tick_interval))
ax.tick_params(axis="both", which="major", labelsize=font_size - 1)

# Labels Below the Plots
fig.text(0.25, -0.05, "(A) ROC Curve", fontsize=font_size + 2, fontweight="bold", ha="center")
fig.text(0.75, -0.05, "(B) P-R Curve", fontsize=font_size + 2, fontweight="bold", ha="center")

# ====================================
# FIGURE 2: TUNED MODELS (C & D)
# ====================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (C) ROC Curve (Tuned Models)
ax = axes[0]
for model, probs in zip(tuned_colors.keys(), [y_probs_wlr_tuned, y_probs_svm_tuned, y_probs_rf_tuned]):
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=f"{model} (AUC: {roc_auc_score(y_test, probs):.3f})", 
            color=tuned_colors[model], linewidth=1.5)

ax.plot([0, 1], [0, 1], linestyle="--", color="red", linewidth=1)  # Diagonal reference line
ax.set_xlabel("False Positive Rate", fontsize=font_size)
ax.set_ylabel("True Positive Rate", fontsize=font_size)
ax.legend(fontsize=legend_size, frameon=True)
ax.set_xticks(np.arange(0, 1.1, axis_tick_interval))
ax.set_yticks(np.arange(0, 1.1, axis_tick_interval))
ax.tick_params(axis="both", which="major", labelsize=font_size - 1)

# (D) Precision-Recall Curve (Tuned Models)
ax = axes[1]
for model, probs in zip(tuned_colors.keys(), [y_probs_wlr_tuned, y_probs_svm_tuned, y_probs_rf_tuned]):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ax.plot(recall, precision, label=model, 
            color=tuned_colors[model], linewidth=1.5)

ax.plot([0, 1], [0.3, 0.3], linestyle="--", color="red", linewidth=1)
ax.set_xlabel("Recall", fontsize=font_size)
ax.set_ylabel("Precision", fontsize=font_size)
ax.legend(fontsize=legend_size, frameon=True, loc="lower right")
ax.set_xticks(np.arange(0, 1.1, axis_tick_interval))
ax.set_yticks(np.arange(0, 1.1, axis_tick_interval))
ax.tick_params(axis="both", which="major", labelsize=font_size - 1)

# Labels Below the Plots
fig.text(0.25, -0.05, "(C) ROC Curve (Tuned)", fontsize=font_size + 2, fontweight="bold", ha="center")
fig.text(0.75, -0.05, "(D) P-R Curve (Tuned)", fontsize=font_size + 2, fontweight="bold", ha="center")

plt.tight_layout()
plt.show()


# ====================================================
# STEP 9: Function to Compute Hidden Positive Rates
# ====================================================
def compute_hidden_positive_rates(y_probs, y_test, thresholds):
    """ Computes hidden positive rate for different probability thresholds. """
    hidden_positives = [np.sum(y_probs[y_test == 0] >= t) for t in thresholds]
    total_unlabeled = np.sum(y_test == 0)
    hidden_positive_rate = [(hp / total_unlabeled) * 100 for hp in hidden_positives]
    return hidden_positive_rate

# Define probability thresholds
thresholds = np.arange(0.5, 0.9, 0.05)

# Compute hidden positive rates for BASE models
hidden_positive_rate_wlr = compute_hidden_positive_rates(y_probs_wlr, y_test, thresholds)
hidden_positive_rate_svm = compute_hidden_positive_rates(y_probs_svm, y_test, thresholds)
hidden_positive_rate_rf = compute_hidden_positive_rates(y_probs_rf, y_test, thresholds)

# Compute hidden positive rates for TUNED models
hidden_positive_rate_wlr_tuned = compute_hidden_positive_rates(y_probs_wlr_tuned, y_test, thresholds)
hidden_positive_rate_svm_tuned = compute_hidden_positive_rates(y_probs_svm_tuned, y_test, thresholds)
hidden_positive_rate_rf_tuned = compute_hidden_positive_rates(y_probs_rf_tuned, y_test, thresholds)

# ================================================
# Function to Plot Hidden Positive Rates
# ================================================
def plot_hidden_positive_rates(hidden_rates, labels, colors, agreement_text, title):
    """ Plots Hidden Positive Rates for given models. """
    plt.figure(figsize=(7, 5))
    for rate, label, color in zip(hidden_rates, labels, colors):
        plt.plot(thresholds, rate, marker='o', linestyle='-', label=label, color=color)
    
    plt.xlabel("Prediction Threshold ($\\theta_T$)", fontsize=12)
    plt.ylabel("Hidden Positives (%)", fontsize=12)
    plt.legend(fontsize=11, frameon=True)
    
    # Display agreement values below the plot
    plt.figtext(0.5, -0.03, agreement_text, ha="center", fontsize=12, fontweight="bold", bbox=dict(facecolor='white', edgecolor='none'))

    # Adjust layout to prevent text cutoff
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.show()

# ========================
# FIGURE 1: BASE MODELS
# ========================
base_labels = ["PU-WLR", "Biased SVM", "PU Bagging"]
base_colors = ["#083864", "#169bbd", "#1c8065"]
base_agreement_text = "IMAR : 0.8165      $\\kappa_F$ : 0.7069      $\\alpha_K$ : 0.7067"

plot_hidden_positive_rates(
    [hidden_positive_rate_wlr, hidden_positive_rate_svm, hidden_positive_rate_rf], 
    base_labels, base_colors, base_agreement_text, "Base Model Hidden Positive Rates"
)

# ========================
# FIGURE 2: TUNED MODELS
# ========================
tuned_labels = ["PU-WLR (Tuned)", "Biased SVM (Tuned)", "PU Bagging (Tuned)"]
tuned_colors = ["#083864", "#169bbd", "#1c8065"]
tuned_agreement_text = "IMAR : 0.8240      $\\kappa_F$ : 0.7182      $\\alpha_K$ : 0.7180"

plot_hidden_positive_rates(
    [hidden_positive_rate_wlr_tuned, hidden_positive_rate_svm_tuned, hidden_positive_rate_rf_tuned], 
    tuned_labels, tuned_colors, tuned_agreement_text, "Tuned Model Hidden Positive Rates"
)


#===========================================
# STEP 00: DISTRIBUTION OF VARS BY HBP RISK
#===========================================

import seaborn as sns

# Load the dataset and drop missing values
data = pd.read_csv("dataset_HBP00.csv").dropna()

# Rename the last column to "HBP Risk"
data.rename(columns={data.columns[-1]: "HBP Risk"}, inplace=True)

# Map values: 1 → "Diagnosed", 0 → "Undiagnosed"
data["HBP Risk"] = data["HBP Risk"].map({1: "Diagnosed", 0: "Undiagnosed"})

# Define custom colors
custom_palette = {"Diagnosed": "#e2b82d", "Undiagnosed": "#1d7e64"}

# Define the number of rows and columns for the grid layout
num_vars = data.shape[1] - 1  # Excluding the target variable
num_cols = 3  # Number of columns in the grid
num_rows = (num_vars + num_cols - 1) // num_cols  # Compute required rows

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
axes = axes.flatten()

# Loop through each variable and plot the distribution
for i, column in enumerate(data.columns[:-1]):  # Exclude the target variable
    sns.kdeplot(
        data=data, x=column, hue="HBP Risk", ax=axes[i], fill=True, alpha=0.5, palette=custom_palette
    )
    axes[i].set_title(f"Distribution of {column}", fontsize=12)
    axes[i].set_xlabel(column, fontsize=10)
    axes[i].set_ylabel("Density", fontsize=10)

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

#==========================================
# SUMMARY STATISTICS
#=========================================

from scipy.stats import skew, kurtosis

# Load the dataset and drop missing values
data = pd.read_csv("dataset_HBP00.csv").dropna()

# Rename the last column to "HBP Risk"
data.rename(columns={data.columns[-1]: "HBP Risk"}, inplace=True)

# Map values: 1 → "Diagnosed", 0 → "Undiagnosed"
data["HBP Risk"] = data["HBP Risk"].map({1: "Diagnosed", 0: "Undiagnosed"})
descriptive_stats = data.describe()

# Compute summary statistics
summary_stats = data.groupby('HBP Risk').describe()
summary_stats.to_csv("HBP_summary_statistics.csv")

# Compute skewness for each group
skewness = data.groupby('HBP Risk').apply(lambda x: x.skew()).T
skewness.to_csv("HBP_skewness.csv")

# Compute kurtosis for each group
kurt = data.groupby('HBP Risk').apply(lambda x: x.kurtosis()).T
kurt.to_csv("HBP_kurtosis.csv")


#==========================================
# CORRELATION 
#=========================================
data = pd.read_csv("dataset_HBP00.csv").dropna()
correlation_matrix = data.corr()

# Set figure size
plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Title and formatting
plt.title("Correlation Matrix of Variables", fontsize=14)
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
plt.yticks(rotation=0)
plt.show()