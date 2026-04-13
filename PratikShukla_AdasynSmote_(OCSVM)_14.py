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
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import os
import time

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

# Create a directory for results
os.makedirs("results", exist_ok=True)
os.makedirs("confusion_matrices", exist_ok=True)

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

# --- Step 3: Preparing for resampling methods ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
resampling_stats = []
timing_results = []

# --- Step 4: Cross-Validation with Resampling for Binary Classification ---
print("\nEvaluating Models with Resampling...")
fold_idx = 1

for train_index, test_index in kf.split(X_pca, y_binary):
    X_train, X_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y_binary[train_index], y_binary[test_index]
    
    print(f"\nProcessing Fold {fold_idx}...")
    
    # Original class distribution
    original_class_dist = pd.Series(y_train).value_counts().to_dict()
    print(f"Original class distribution: {original_class_dist}")
    
    # Multi-stage resampling approach for this fold
    try:
        # Stage 1: Use RandomOverSampler to boost very rare classes
        min_samples_needed = 10  # Minimum samples needed for SMOTE/ADASYN
        rare_classes = []
        class_counts = pd.Series(y_train).value_counts()
        
        for cls, count in class_counts.items():
            if count < min_samples_needed:
                rare_classes.append(cls)
                
        if rare_classes:
            print(f"Classes with fewer than {min_samples_needed} samples: {rare_classes}")
            # Set sampling strategy to increase rare classes
            sampling_strategy = {cls: max(min_samples_needed, count) for cls, count in class_counts.items() if count < min_samples_needed}
            if sampling_strategy:
                ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
                ros_class_dist = pd.Series(y_train_ros).value_counts().to_dict()
                print(f"After RandomOverSampler: {ros_class_dist}")
            else:
                X_train_ros, y_train_ros = X_train, y_train
        else:
            X_train_ros, y_train_ros = X_train, y_train
            
        # Stage 2: Apply ADASYN and SMOTE
        # For ADASYN
        adasyn_success = False
        try:
            start_time = time.time()
            adasyn = ADASYN(sampling_strategy='auto', n_neighbors=5, random_state=42)
            X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_ros, y_train_ros)
            adasyn_time = time.time() - start_time
            adasyn_class_dist = pd.Series(y_train_adasyn).value_counts().to_dict()
            print(f"After ADASYN: {adasyn_class_dist}")
            adasyn_success = True
        except Exception as e:
            print(f"ADASYN failed: {str(e)}")
            X_train_adasyn, y_train_adasyn = X_train_ros, y_train_ros
            adasyn_time = 0
            
        # For SMOTE
        smote_success = False
        try:
            start_time = time.time()
            smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_ros, y_train_ros)
            smote_time = time.time() - start_time
            smote_class_dist = pd.Series(y_train_smote).value_counts().to_dict()
            print(f"After SMOTE: {smote_class_dist}")
            smote_success = True
        except Exception as e:
            print(f"SMOTE failed: {str(e)}")
            X_train_smote, y_train_smote = X_train_ros, y_train_ros
            smote_time = 0
            
        # Stage 3: Choose final training set based on results
        if adasyn_success and smote_success:
            print("Using combined ADASYN and SMOTE samples...")
            # Combine ADASYN and SMOTE results (50-50 mix)
            all_classes = np.unique(np.concatenate([y_train_adasyn, y_train_smote]))
            X_final = []
            y_final = []
            
            for cls in all_classes:
                adasyn_idx = np.where(y_train_adasyn == cls)[0]
                smote_idx = np.where(y_train_smote == cls)[0]
                
                # Calculate samples to take from each source
                total_samples = len(adasyn_idx) + len(smote_idx)
                adasyn_samples = len(adasyn_idx) // 2
                smote_samples = len(smote_idx) // 2
                
                # Take samples from ADASYN
                if adasyn_samples > 0:
                    selected_adasyn = np.random.choice(adasyn_idx, adasyn_samples, replace=False)
                    X_final.extend(X_train_adasyn[selected_adasyn])
                    y_final.extend(y_train_adasyn[selected_adasyn])
                
                # Take samples from SMOTE
                if smote_samples > 0:
                    selected_smote = np.random.choice(smote_idx, smote_samples, replace=False)
                    X_final.extend(X_train_smote[selected_smote])
                    y_final.extend(y_train_smote[selected_smote])
            
            X_train_final = np.array(X_final)
            y_train_final = np.array(y_final)
            resampling_method = "Combined ADASYN+SMOTE"
            resampling_time = adasyn_time + smote_time
            
        elif adasyn_success:
            print("Using only ADASYN samples...")
            X_train_final, y_train_final = X_train_adasyn, y_train_adasyn
            resampling_method = "ADASYN"
            resampling_time = adasyn_time
            
        elif smote_success:
            print("Using only SMOTE samples...")
            X_train_final, y_train_final = X_train_smote, y_train_smote
            resampling_method = "SMOTE"
            resampling_time = smote_time
            
        else:
            print("Using RandomOverSampler results...")
            X_train_final, y_train_final = X_train_ros, y_train_ros
            resampling_method = "RandomOverSampler"
            resampling_time = 0
            
        final_class_dist = pd.Series(y_train_final).value_counts().to_dict()
        print(f"Final resampled distribution: {final_class_dist}")
        
    except Exception as e:
        print(f"Resampling error: {str(e)}")
        print("Using original data without resampling")
        X_train_final, y_train_final = X_train, y_train
        resampling_method = "None (Original)"
        resampling_time = 0
        final_class_dist = original_class_dist
        
    # Record resampling statistics
    resampling_stats.append({
        'Fold': fold_idx,
        'Original Distribution': str(original_class_dist),
        'Final Distribution': str(final_class_dist),
        'Resampling Method': resampling_method
    })
    
    # --- Step 5: Train the binary classification models ---
    
    # Create base classifiers
    base_classifiers = [
        ('RF', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('GB', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42))
    ]
    
    # Create stacking model
    stacking_model = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=LogisticRegression(max_iter=500),
        passthrough=True  # Pass original features to final estimator
    )
    
    # Train Stacking Model with resampled data
    start_time = time.time()
    stacking_model.fit(X_train_final, y_train_final)
    train_time = time.time() - start_time
    
    # Train OC-SVM with only inlier data
    start_time = time.time()
    # For OC-SVM, use only inlier samples for training
    inlier_indices = np.where(y_train_final == 1)[0]
    best_ocsvm.fit(X_train_final[inlier_indices])
    ocsvm_train_time = time.time() - start_time
    
    # --- Step 6: Evaluate both models on test set ---
    
    # Evaluate Stacking Model
    start_time = time.time()
    y_pred_stacking = stacking_model.predict(X_test)
    stacking_test_time = time.time() - start_time
    
    # Evaluate OC-SVM
    start_time = time.time()
    y_pred_ocsvm = best_ocsvm.predict(X_test)
    ocsvm_test_time = time.time() - start_time
    
    # Record timing
    timing_results.append({
        'Fold': fold_idx,
        'Model': 'Stacking with ' + resampling_method,
        'Training Time (s)': train_time,
        'Testing Time (s)': stacking_test_time,
        'Total Time (s)': train_time + stacking_test_time
    })
    
    timing_results.append({
        'Fold': fold_idx,
        'Model': 'OC-SVM with ' + resampling_method,
        'Training Time (s)': ocsvm_train_time,
        'Testing Time (s)': ocsvm_test_time,
        'Total Time (s)': ocsvm_train_time + ocsvm_test_time
    })
    
    # --- Step 7: Calculate metrics for both models ---
    
    # For Stacking Model
    cm_stacking = confusion_matrix(y_test, y_pred_stacking, labels=[1, -1])
    accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
    precision_stacking = precision_score(y_test, y_pred_stacking, pos_label=1, zero_division=0)
    recall_stacking = recall_score(y_test, y_pred_stacking, pos_label=1)
    f1_stacking = f1_score(y_test, y_pred_stacking, pos_label=1)
    
    # Calculate AUC if predict_proba is available
    try:
        auc_stacking = roc_auc_score((y_test == 1).astype(int), stacking_model.predict_proba(X_test)[:, 1])
    except:
        auc_stacking = np.nan
    
    # For OC-SVM
    cm_ocsvm = confusion_matrix(y_test, y_pred_ocsvm, labels=[1, -1])
    accuracy_ocsvm = accuracy_score(y_test, y_pred_ocsvm)
    precision_ocsvm = precision_score(y_test, y_pred_ocsvm, pos_label=1, zero_division=0)
    recall_ocsvm = recall_score(y_test, y_pred_ocsvm, pos_label=1)
    f1_ocsvm = f1_score(y_test, y_pred_ocsvm, pos_label=1)
    
    # Store results
    results.append({
        'Fold': fold_idx,
        'Model': 'Stacking with ' + resampling_method,
        'Accuracy': accuracy_stacking,
        'Precision': precision_stacking,
        'Recall': recall_stacking,
        'F1 Score': f1_stacking,
        'AUC Score': auc_stacking
    })
    
    results.append({
        'Fold': fold_idx,
        'Model': 'OC-SVM with ' + resampling_method,
        'Accuracy': accuracy_ocsvm,
        'Precision': precision_ocsvm,
        'Recall': recall_ocsvm,
        'F1 Score': f1_ocsvm,
        'AUC Score': np.nan  # OC-SVM doesn't have predict_proba
    })
    
    # --- Step 8: Plot confusion matrices ---
    
    # For Stacking Model
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues', xticklabels=['Inlier', 'Outlier'], yticklabels=['Inlier', 'Outlier'])
    plt.title(f"Stacking with {resampling_method} - Fold {fold_idx} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrices/stacking_{resampling_method.replace(' ', '_').lower()}_fold_{fold_idx}.png")
    plt.close()
    
    # For OC-SVM
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_ocsvm, annot=True, fmt='d', cmap='Blues', xticklabels=['Inlier', 'Outlier'], yticklabels=['Inlier', 'Outlier'])
    plt.title(f"OC-SVM with {resampling_method} - Fold {fold_idx} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrices/ocsvm_{resampling_method.replace(' ', '_').lower()}_fold_{fold_idx}.png")
    plt.close()
    
    # Print results for this fold
    print(f"\nStacking Model with {resampling_method} - Fold {fold_idx}:")
    print(f"Accuracy: {accuracy_stacking:.4f}, Precision: {precision_stacking:.4f}, Recall: {recall_stacking:.4f}, F1 Score: {f1_stacking:.4f}")
    
    print(f"OC-SVM with {resampling_method} - Fold {fold_idx}:")
    print(f"Accuracy: {accuracy_ocsvm:.4f}, Precision: {precision_ocsvm:.4f}, Recall: {recall_ocsvm:.4f}, F1 Score: {f1_ocsvm:.4f}")
    
    fold_idx += 1

# --- Step 9: Save and display results ---

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("results/combined_model_results.csv", index=False)

timing_df = pd.DataFrame(timing_results)
timing_df.to_csv("results/timing_results.csv", index=False)

resampling_df = pd.DataFrame(resampling_stats)
resampling_df.to_csv("results/resampling_statistics.csv", index=False)

# Calculate average metrics by model
print("\nAverage Metrics by Model:")
avg_metrics = results_df.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].mean()
print(avg_metrics)

# Create summary plot
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
avg_metrics_plot = avg_metrics.reset_index()

# Create grouped bar plot
for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.barplot(x='Model', y=metric, data=avg_metrics_plot)
    plt.title(f"Average {metric} by Model")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

plt.savefig("results/average_metrics_comparison.png")
plt.close()

print("\nAnalysis complete. Results saved to the 'results' directory.")
