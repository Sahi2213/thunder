import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import ADASYN, SMOTE
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

# Convert to binary classification problem (1 for inliers, 0 for outliers)
y_train_binary = np.where(y_train == inlier_class, 1, 0)
y_test_binary = np.where(y_test == inlier_class, 1, 0)

# --- Step 6: Apply Resampling Techniques ---
print("Applying resampling techniques to balance the dataset...")

# Initialize dictionaries to store resampled datasets
resampling_methods = {
    'original': {'X': X_train, 'y': y_train_binary},
    'smote': {},
    'adasyn': {}
}

# Check if minority class has enough samples for SMOTE/ADASYN
minority_count = min(np.sum(y_train_binary == 0), np.sum(y_train_binary == 1))
sampling_strategy = 0.8  # 80% of majority class

if minority_count >= 6:  # Both SMOTE and ADASYN require at least 6 samples
    # Apply SMOTE
    print("Applying SMOTE resampling...")
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train_binary)
    resampling_methods['smote'] = {'X': X_train_smote, 'y': y_train_smote}
    print(f"Original dataset shape: {np.bincount(y_train_binary)}")
    print(f"SMOTE resampled dataset shape: {np.bincount(y_train_smote)}")
    
    # Apply ADASYN
    print("Applying ADASYN resampling...")
    adasyn = ADASYN(random_state=42, sampling_strategy=sampling_strategy)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train_binary)
    resampling_methods['adasyn'] = {'X': X_train_adasyn, 'y': y_train_adasyn}
    print(f"ADASYN resampled dataset shape: {np.bincount(y_train_adasyn)}")
else:
    print(f"Warning: Minority class has only {minority_count} samples, which is less than the minimum required for SMOTE/ADASYN.")
    print("Skipping resampling and using original dataset.")

# --- Step 7: Find Best OSVM Parameters ---
print("Finding best OSVM parameters using original dataset...")
# Extract inliers for OSVM training
X_train_inliers_original = X_train[y_train_binary == 1]

# Remove potential outliers using Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
iso_pred = iso.fit_predict(X_train_inliers_original)
X_train_clean_original = X_train_inliers_original[iso_pred == 1]
print(f"Training with {X_train_clean_original.shape[0]} clean inliers out of {X_train_inliers_original.shape[0]} initial inliers.")

# Train OSVM with hyperparameter tuning
param_grid = {
    'nu': [0.01, 0.05, 0.1],
    'gamma': [0.0001, 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}
osvm = OneClassSVM()
grid_search = GridSearchCV(
    osvm, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1
)
grid_search.fit(X_train_clean_original)
best_params = grid_search.best_params_
print(f"Best OSVM Parameters: {best_params}")

# --- Step 8: Train OSVM with Different Resampling Methods ---
print("Training OSVM models with different resampling methods...")
osvm_models = {}
thresholds = {}
results = {}

for method_name, data_dict in resampling_methods.items():
    print(f"\nTraining OSVM with {method_name} data...")
    X_train_method = data_dict['X']
    y_train_method = data_dict['y']
    
    # Extract inliers for OSVM training
    X_train_inliers = X_train_method[y_train_method == 1]
    
    # Remove potential outliers using Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = iso.fit_predict(X_train_inliers)
    X_train_clean = X_train_inliers[iso_pred == 1]
    print(f"Training with {X_train_clean.shape[0]} clean inliers out of {X_train_inliers.shape[0]} initial inliers.")
    
    # Train OSVM with best parameters found earlier
    osvm_model = OneClassSVM(**best_params)
    osvm_model.fit(X_train_clean)
    osvm_models[method_name] = osvm_model
    
    # Calibrate decision threshold
    decision_scores_train = osvm_model.decision_function(X_train_method)
    y_train_osvm_format = np.where(y_train_method == 1, 1, -1)
    
    fpr, tpr, thresholds_roc = roc_curve(y_train_osvm_format, decision_scores_train)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds_roc[optimal_idx]
    thresholds[method_name] = optimal_threshold
    roc_auc_value = auc(fpr, tpr)
    
    print(f"Optimal Threshold: {optimal_threshold:.4f} | AUC: {roc_auc_value:.4f}")
    
    # Evaluate on test set
    decision_scores = osvm_model.decision_function(X_test)
    y_pred_osvm = np.where(decision_scores < optimal_threshold, -1, 1)
    y_test_osvm_format = np.where(y_test_binary == 1, 1, -1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_osvm_format, y_pred_osvm)
    report = classification_report(y_test_osvm_format, y_pred_osvm, output_dict=True)
    conf_mat = confusion_matrix(y_test_osvm_format, y_pred_osvm)
    
    results[method_name] = {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': conf_mat
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_osvm_format, y_pred_osvm))

# --- Step 9: Compare Results ---
print("\n--- Comparison of Different Resampling Methods ---")
accuracies = {method: result['accuracy'] for method, result in results.items()}
print("Accuracy Comparison:")
for method, acc in accuracies.items():
    print(f"{method.capitalize()}: {acc:.4f}")

# --- Step 10: Plot Confusion Matrices ---
plt.figure(figsize=(15, 5))
for i, (method, result) in enumerate(results.items(), 1):
    plt.subplot(1, len(results), i)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Outliers", "Inliers"],
                yticklabels=["Outliers", "Inliers"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{method.capitalize()} Confusion Matrix")
plt.tight_layout()
plt.show()

# --- Step 11: Plot ROC Curves ---
plt.figure(figsize=(10, 8))
for method_name, data_dict in resampling_methods.items():
    osvm_model = osvm_models[method_name]
    y_train_method = data_dict['y']
    y_train_osvm_format = np.where(y_train_method == 1, 1, -1)
    
    # Calculate ROC curve
    decision_scores = osvm_model.decision_function(X_test)
    y_test_osvm_format = np.where(y_test_binary == 1, 1, -1)
    fpr, tpr, _ = roc_curve(y_test_osvm_format, decision_scores)
    roc_auc_value = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{method_name.capitalize()} (AUC = {roc_auc_value:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Resampling Methods')
plt.legend(loc="lower right")
plt.show()

# --- Step 12: Save Best Model ---
best_method = max(accuracies, key=accuracies.get)
print(f"\nBest performing method: {best_method.capitalize()} with accuracy {accuracies[best_method]:.4f}")
print(f"Saving {best_method} model and preprocessing components...")

joblib.dump(osvm_models[best_method], f'osvm_model_{best_method}.pkl')
joblib.dump(scaler, 'robust_scaler.pkl')
joblib.dump(ipca, 'incremental_pca.pkl')
joblib.dump(thresholds[best_method], f'osvm_threshold_{best_method}.pkl')
print("Models saved successfully!")

# Optional: Save all models
print("Also saving all models for future comparison...")
for method, model in osvm_models.items():
    joblib.dump(model, f'osvm_model_{method}.pkl')
    joblib.dump(thresholds[method], f'threshold_{method}.pkl')
print("All models saved!")
