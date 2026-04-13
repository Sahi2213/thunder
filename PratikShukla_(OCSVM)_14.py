import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
import os

# --- Step 1: Load and Preprocess Dataset ---
print("Loading dataset...")
df = pd.read_csv('part-00001_preprocessed_dataset.csv')

# Rename the last column as 'label'
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

# Print class distribution
print(f"Class distribution:\n{df['label'].value_counts()}")

# Features and labels
X = df.drop(columns=['label']).values
y = df['label'].values

# Define inliers (majority class) and outliers
majority_class = 4.0
y_binary = np.where(y == majority_class, 1, -1)  # 1: Inliers, -1: Outliers

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)
print(f"Reduced dimensions from {X.shape[1]} to {X_pca.shape[1]}")

# --- Step 2: OC-SVM Tuning and Training ---
print("\nTuning OC-SVM...")
ocsvm_param_grid = {
    'nu': [0.01, 0.05, 0.1],
    'gamma': ['scale', 0.1, 1],
    'kernel': ['rbf', 'sigmoid']  # Expanded kernels for better exploration
}

grid_search = GridSearchCV(
    OneClassSVM(), ocsvm_param_grid,
    cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_pca[y_binary == 1])  # Train only on inliers
best_ocsvm = grid_search.best_estimator_
print(f"\nBest OC-SVM Hyperparameters: {grid_search.best_params_}")

# --- Step 3: Stacked Ensemble Model ---
base_classifiers = [
    ('RF', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('GB', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))
]
stacking_model = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=LogisticRegression(max_iter=500),
    passthrough=True  # Pass original features and transformed features to final estimator
)

# --- Step 4: Cross-Validation and Model Evaluation ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# Create directories for saving confusion matrices if not existing
os.makedirs("confusion_matrices", exist_ok=True)

print("\nEvaluating Stacked Model...")
fold_idx = 1
for train_index, test_index in kf.split(X_pca, y_binary):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y_binary[train_index], y_binary[test_index]

    # Train Stacking Model
    stacking_model.fit(X_train, y_train)

    # Predict
    y_pred = stacking_model.predict(X_test)

    # Compute metrics
    cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score((y_test == 1).astype(int), stacking_model.predict_proba(X_test)[:, 1])

    # Store results for this fold
    results.append({
        'Fold': fold_idx,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC Score': auc
    })

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Inlier', 'Outlier'], yticklabels=['Inlier', 'Outlier'])
    plt.title(f"Stacked Model - Fold {fold_idx} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrices/stacked_model_fold_{fold_idx}.png")
    plt.close()

    print(f"Fold {fold_idx} Results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}, AUC={auc:.4f}")
    fold_idx += 1

# Summarize results
results_df = pd.DataFrame(results)
print("\nCross-Validation Results:")
print(results_df)

# Save results to CSV
results_df.to_csv("stacked_model_results.csv", index=False)

# --- Step 5: Independent OC-SVM Evaluation ---
print("\nEvaluating OC-SVM Independently...")
ocsvm_results = []
fold_idx = 1
for train_index, test_index in kf.split(X_pca, y_binary):  # Pass both X_pca and y_binary
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y_binary[train_index], y_binary[test_index]

    # Train OC-SVM on inliers
    best_ocsvm.fit(X_train[y_train == 1])

    # Predict
    y_pred = best_ocsvm.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Store results
    ocsvm_results.append({
        'Fold': fold_idx,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

    print(f"OC-SVM Fold {fold_idx} Results: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
    fold_idx += 1

# Summarize OC-SVM results
ocsvm_results_df = pd.DataFrame(ocsvm_results)
print("\nOC-SVM Cross-Validation Results:")
print(ocsvm_results_df)

# Save results to CSV
ocsvm_results_df.to_csv("ocsvm_results.csv", index=False)
