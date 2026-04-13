import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler

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

# --- Step 2: Layer 1 - Anomaly Detection ---
print("Applying Layer 1: Anomaly Detection with Isolation Forest...")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)

# Retain only inliers for supervised classification
X_filtered = X_scaled[anomaly_labels == 1]
y_filtered = y_encoded[anomaly_labels == 1]
print(f"Data after anomaly detection: {X_filtered.shape[0]} samples")

# --- Step 3: Handle Class Imbalance ---
print("Balancing classes with SMOTE...")
try:
    smote = SMOTE(random_state=42, k_neighbors=1)  # Adjust k_neighbors for rare classes
    X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)
except ValueError as e:
    print(f"SMOTE failed due to rare classes: {e}")
    print("Falling back to RandomOverSampler...")
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_filtered, y_filtered)

# Ensure consistent mapping post-resampling
unique_classes = np.unique(y_resampled)
if not np.array_equal(unique_classes, np.arange(len(unique_classes))):
    print("Detected non-consecutive labels post-resampling. Re-encoding...")
    y_resampled = LabelEncoder().fit_transform(y_resampled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# --- Step 4: Layer 2 - Supervised Classification ---
print("Training Layer 2: Supervised Classification with Random Forest and XGBoost...")
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

# --- Step 5: Combine Predictions Using Weighted Voting ---
print("Combining predictions with weighted voting...")
rf_preds = rf_best_model.predict(X_test)
xgb_preds = xgb_best_model.predict(X_test)

# Weighted voting
rf_weight = 0.4
xgb_weight = 0.6
final_preds = np.round(rf_preds * rf_weight + xgb_preds * xgb_weight).astype(int)

# Decode predictions back to original labels
final_preds_decoded = le.inverse_transform(final_preds)

# --- Step 6: Evaluate Performance ---
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

# --- Step 7: Save Evaluation Results ---
classification_rep = classification_report(le.inverse_transform(y_test), final_preds_decoded, output_dict=True)
pd.DataFrame(classification_rep).to_csv("classification_report.csv")
pd.DataFrame(confusion_matrix(le.inverse_transform(y_test), final_preds_decoded)).to_csv("confusion_matrix.csv")
