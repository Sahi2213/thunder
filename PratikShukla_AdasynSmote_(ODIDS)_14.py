import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler

# --- Step 1: Load and Preprocess Dataset ---
print("Loading and preprocessing dataset...")
data = pd.read_csv("part-00001_preprocessed_dataset.csv")  # Replace path with your dataset

# Check for missing values and impute median values for numeric features
data.fillna(data.median(numeric_only=True), inplace=True)

# Define target column
target_col = "label"  # Adjust to match your dataset's target variable
if target_col not in data.columns:
    raise KeyError(f"Target column '{target_col}' not found in the dataset.")

# Features and target
X = data.drop(columns=[target_col])
y = data[target_col]

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode target labels (force consistent mapping to consecutive integers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Transform labels to consecutive integers
print(f"Original Labels: {list(le.classes_)}")
print(f"Encoded Labels: {np.unique(y_encoded)}")

# --- Step 2: Handle Class Imbalance with ADASYN and SMOTE ---
print("Balancing classes with ADASYN and SMOTE...")

# Find the minimum class size to ensure n_neighbors doesn't exceed it
min_class_size = min(np.bincount(y_encoded))
n_neighbors = min(5, min_class_size - 1)  # Make sure n_neighbors doesn't exceed the number of samples in the smallest class

# Apply ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(sampling_strategy='auto', n_neighbors=n_neighbors, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y_encoded)
print(f"After ADASYN resampling: {np.unique(y_resampled, return_counts=True)}")

# Apply SMOTE for additional resampling
smote = SMOTE(sampling_strategy='auto', k_neighbors=n_neighbors, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)
print(f"After SMOTE resampling: {np.unique(y_resampled, return_counts=True)}")

# Ensure consistent mapping post-resampling
unique_classes = np.unique(y_resampled)
if not np.array_equal(unique_classes, np.arange(len(unique_classes))):
    print("Detected non-consecutive labels post-resampling. Re-encoding...")
    y_resampled = LabelEncoder().fit_transform(y_resampled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# --- Step 3: Supervised Classification with Random Forest and XGBoost ---
print("Training classifiers (Random Forest and XGBoost)...")
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf_model, rf_params, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
rf_best_model = rf_grid.best_estimator_

# Hyperparameter tuning for XGBoost
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.2]
}
xgb_grid = GridSearchCV(xgb_model, xgb_params, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
xgb_grid.fit(X_train, y_train)
xgb_best_model = xgb_grid.best_estimator_

# --- Step 4: Combine Predictions Using Weighted Voting ---
print("Combining predictions with weighted voting...")
rf_preds = rf_best_model.predict(X_test)
xgb_preds = xgb_best_model.predict(X_test)

# Weighted voting
rf_weight = 0.4
xgb_weight = 0.6
final_preds = np.round(rf_preds * rf_weight + xgb_preds * xgb_weight).astype(int)

# Decode predictions back to original labels
final_preds_decoded = le.inverse_transform(final_preds)

# --- Step 5: Evaluate Performance ---
accuracy = accuracy_score(le.inverse_transform(y_test), final_preds_decoded)
print(f"Optimized ODIDS Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(le.inverse_transform(y_test), final_preds_decoded))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(le.inverse_transform(y_test), final_preds_decoded), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Optimized ODIDS")
plt.show()

# --- Step 6: Save Evaluation Results ---
classification_rep = classification_report(le.inverse_transform(y_test), final_preds_decoded, output_dict=True)
pd.DataFrame(classification_rep).to_csv("classification_report.csv")
pd.DataFrame(confusion_matrix(le.inverse_transform(y_test), final_preds_decoded)).to_csv("confusion_matrix.csv")
