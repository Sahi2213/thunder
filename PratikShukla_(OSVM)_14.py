import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Step 1: Load and Preprocess Dataset ---
print("Loading and preprocessing dataset...")
data = pd.read_csv("part-00001_preprocessed_dataset.csv")  # Replace with your dataset path

# Define target column
target_col = "label"  # Adjust to match your dataset's target variable
if target_col not in data.columns:
    raise KeyError(f"Target column '{target_col}' not found in the dataset.")

# Handle missing values
data.fillna(data.median(numeric_only=True), inplace=True)

# Split features and target
X = data.drop(columns=[target_col])  # Features
y = data[target_col]

# Encode categorical features
categorical_features = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# --- Step 2: Feature Selection Using Mutual Information ---
print("Selecting most informative features...")
k = min(50, X.shape[1])  # Select top k features or all if fewer than k
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
top_features = mi_df.sort_values('MI_Score', ascending=False).head(k)['Feature'].tolist()
X_selected = X[top_features]

# --- Step 3: Apply Robust Scaling ---
print("Scaling selected features...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

# --- Step 4: Apply Incremental PCA ---
print("Applying Incremental PCA...")
ipca = IncrementalPCA(n_components=min(20, X_scaled.shape[1]), batch_size=1000)
X_kpca = ipca.fit_transform(X_scaled)
print("Incremental PCA completed successfully.")

# Define inlier class (adjust based on dataset)
inlier_class = 4.0

# --- Step 5: Split Data for Training and Testing ---
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_kpca, y, test_size=0.2, random_state=42, stratify=y
)

# Extract inliers for OSVM training
X_train_inliers = X_train[y_train == inlier_class]

# --- Step 6: Remove Potential Outliers Using Isolation Forest ---
print("Removing outliers from training data using Isolation Forest...")
iso = IsolationForest(contamination=0.05, random_state=42)
iso_pred = iso.fit_predict(X_train_inliers)
X_train_clean = X_train_inliers[iso_pred == 1]
print(f"Training with {X_train_clean.shape[0]} clean inliers out of {X_train_inliers.shape[0]} initial inliers.")

# --- Step 7: Train OSVM with Hyperparameter Tuning ---
print("Training One-Class SVM with hyperparameter tuning...")
param_grid = {
    'nu': [0.01, 0.05, 0.1],
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}
osvm = OneClassSVM()
grid_search = GridSearchCV(
    osvm, param_grid, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1
)
grid_search.fit(X_train_clean)
best_osvm = grid_search.best_estimator_
print(f"Best OSVM Parameters: {grid_search.best_params_}")

# --- Step 8: Calibrate Decision Threshold ---
print("Calibrating decision threshold...")
decision_scores_train = best_osvm.decision_function(X_train)
y_train_binary = np.where(y_train == inlier_class, 1, -1)

fpr, tpr, thresholds = roc_curve(y_train_binary, decision_scores_train)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
roc_auc = auc(fpr, tpr)
print(f"Optimal Threshold: {optimal_threshold:.4f} | AUC: {roc_auc:.4f}")

# --- Step 9: Predict and Evaluate ---
print("Evaluating OSVM...")
decision_scores = best_osvm.decision_function(X_test)
y_pred_osvm = np.where(decision_scores < optimal_threshold, -1, 1)
y_test_binary = np.where(y_test == inlier_class, 1, -1)

# Accuracy and Classification Report
accuracy = accuracy_score(y_test_binary, y_pred_osvm)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_osvm))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_binary, y_pred_osvm)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Outliers", "Inliers"],
            yticklabels=["Outliers", "Inliers"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("OSVM Confusion Matrix")
plt.tight_layout()
plt.show()

# --- Step 10: Save Model ---
print("Saving model and preprocessing components...")
joblib.dump(best_osvm, 'osvm_model.pkl')
joblib.dump(scaler, 'robust_scaler.pkl')
joblib.dump(ipca, 'incremental_pca.pkl')
print("Model saved successfully!")
