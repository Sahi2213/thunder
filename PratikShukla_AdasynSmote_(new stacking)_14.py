import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix)
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import time
import os

# Read the dataset
df = pd.read_csv('part-00001_preprocessed_dataset.csv')

# Take 20% of the data
df = df.sample(frac=0.2, random_state=42)

# Rename the last column as 'label'
df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

X = df.drop(columns=['label']).values
y = df['label'].values

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define Stratified K-Folds cross-validation
K_FOLDS = 5
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

# Create a directory for confusion matrices
os.makedirs("confusion_matrices", exist_ok=True)

# Define base classifiers
base_classifiers = [
    ('DecisionTree', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('RandomForest', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ('KNN', KNeighborsClassifier(n_neighbors=7, weights='distance')),
    ('LogisticRegression', LogisticRegression(max_iter=500, random_state=42))
]

# Define meta-classifier
meta_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Define Stacking model
stacking_model = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=meta_classifier,
    passthrough=True  # Include original features in meta-classifier training
)

# Store results
results = []
timing_results = []
resampling_stats = []

# Iterate over Stratified K-Folds
fold_idx = 1
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f"Processing Fold {fold_idx}...")
    
    # Get class distribution before resampling
    original_class_dist = pd.Series(y_train).value_counts().to_dict()
    print(f"Original class distribution in fold {fold_idx}: {original_class_dist}")
    
    # Count class occurrences
    class_counts = pd.Series(y_train).value_counts()
    singleton_classes = class_counts[class_counts == 1].index.tolist()
    low_count_classes = class_counts[class_counts < 6].index.tolist()
    
    print(f"Classes with only one instance: {singleton_classes}")
    print(f"Classes with less than 6 instances: {low_count_classes}")
    
    # Two-stage resampling process
    try:
        # STAGE 1: Use RandomOverSampler for classes with very few samples
        print("Stage 1: Using RandomOverSampler for rare classes...")
        # Set sampling_strategy to increase only classes with < 6 samples to exactly 6 samples
        sampling_strategy = {cls: max(6, count) for cls, count in class_counts.items() if count < 6}
        if sampling_strategy:
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
            ros_class_dist = pd.Series(y_train_ros).value_counts().to_dict()
            print(f"After RandomOverSampler: {ros_class_dist}")
        else:
            X_train_ros, y_train_ros = X_train, y_train
            
        # STAGE 2: Now apply ADASYN and SMOTE on data with sufficient samples per class
        # For ADASYN
        adasyn_sampling_strategy = 'auto'  # 'auto' will use all classes with samples > 6
        adasyn = ADASYN(sampling_strategy=adasyn_sampling_strategy, n_neighbors=5, random_state=42)
        try:
            X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_ros, y_train_ros)
            adasyn_class_dist = pd.Series(y_train_adasyn).value_counts().to_dict()
            print(f"After ADASYN: {adasyn_class_dist}")
            adasyn_success = True
        except Exception as e:
            print(f"ADASYN failed: {e}")
            X_train_adasyn, y_train_adasyn = X_train_ros, y_train_ros
            adasyn_success = False
        
        # For SMOTE
        smote_sampling_strategy = 'auto'  # 'auto' will use all classes with samples > 5
        smote = SMOTE(sampling_strategy=smote_sampling_strategy, k_neighbors=5, random_state=42)
        try:
            X_train_smote, y_train_smote = smote.fit_resample(X_train_ros, y_train_ros)
            smote_class_dist = pd.Series(y_train_smote).value_counts().to_dict()
            print(f"After SMOTE: {smote_class_dist}")
            smote_success = True
        except Exception as e:
            print(f"SMOTE failed: {e}")
            X_train_smote, y_train_smote = X_train_ros, y_train_ros
            smote_success = False
            
        # STAGE 3: Combine the results based on which algorithms succeeded
        if adasyn_success and smote_success:
            print("Using mixed ADASYN and SMOTE samples...")
            # Combine samples from both methods (50-50 split for each class)
            all_classes = np.unique(np.concatenate([y_train_adasyn, y_train_smote]))
            X_final = []
            y_final = []
            
            for cls in all_classes:
                adasyn_idx = np.where(y_train_adasyn == cls)[0]
                smote_idx = np.where(y_train_smote == cls)[0]
                
                # Calculate how many samples to take from each source
                total_samples = max(len(adasyn_idx), len(smote_idx))
                adasyn_samples = min(len(adasyn_idx), total_samples // 2 + (total_samples % 2))
                smote_samples = min(len(smote_idx), total_samples - adasyn_samples)
                
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
        elif adasyn_success:
            print("Using only ADASYN samples...")
            X_train_final, y_train_final = X_train_adasyn, y_train_adasyn
        elif smote_success:
            print("Using only SMOTE samples...")
            X_train_final, y_train_final = X_train_smote, y_train_smote
        else:
            print("Using randomly oversampled data...")
            X_train_final, y_train_final = X_train_ros, y_train_ros
            
        # Record final distribution
        final_class_dist = pd.Series(y_train_final).value_counts().to_dict()
        print(f"Final training distribution: {final_class_dist}")
        
        # Record resampling statistics
        resampling_stats.append({
            'Fold': fold_idx,
            'Original_Distribution': original_class_dist,
            'Final_Distribution': final_class_dist
        })
        
    except Exception as e:
        print(f"Resampling error on Fold {fold_idx}: {e}")
        print("Using original data without resampling")
        X_train_final, y_train_final = X_train, y_train

    # Train the Stacking model
    start_train_time = time.time()
    stacking_model.fit(X_train_final, y_train_final)
    train_time = time.time() - start_train_time

    # Test the Stacking model
    start_test_time = time.time()
    y_pred = stacking_model.predict(X_test)
    test_time = time.time() - start_test_time

    # Record timing
    timing_results.append({
        'Fold': fold_idx,
        'Classifier': 'Stacking with Combined Resampling',
        'Training Time (s)': train_time,
        'Testing Time (s)': test_time,
        'Total Time (s)': train_time + test_time
    })

    # Compute metrics
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

    results.append({
        'Fold': fold_idx,
        'Classifier': 'Stacking with Combined Resampling',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    })

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
    plt.title(f"Stacking with Combined Resampling - Fold {fold_idx} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrices/StackingModel_Combined_Resampling_fold_{fold_idx}.png")
    plt.close()

    fold_idx += 1

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("stacking_combined_resampling_metrics.csv", index=False)

timing_df = pd.DataFrame(timing_results)
timing_df.to_csv("stacking_combined_resampling_time.csv", index=False)

# Save resampling statistics
resampling_df = pd.DataFrame(resampling_stats)
resampling_df.to_csv("resampling_statistics.csv", index=False)

# Calculate and print average metrics
print("\nAverage Classification Metrics Across Folds:")
average_metrics = results_df.drop('Fold', axis=1).groupby('Classifier').mean()
print(average_metrics)

print("\nClassification Metrics by Fold:")
print(results_df)
