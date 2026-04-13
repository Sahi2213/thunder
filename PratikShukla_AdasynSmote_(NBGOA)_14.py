import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore")

# --- ADASYN Implementation ---
class ADASYN:
    def __init__(self, beta=1.0, k_neighbors=5, random_state=None):
        """
        Parameters:
        -----------
        beta : float, optional (default=1.0)
            Specifies the desired balance level after generation. 
            beta=1.0 means fully balanced classes.
        
        k_neighbors : int, optional (default=5)
            Number of nearest neighbors to use when determining density.
        
        random_state : int, optional (default=None)
            Controls the randomization of the algorithm.
        """
        self.beta = beta
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit_resample(self, X, y):
        """
        Applies ADASYN algorithm to balance the dataset.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Features of the dataset.
        
        y : array-like, shape (n_samples,)
            Target variable of the dataset.
            
        Returns:
        --------
        X_resampled : array-like, shape (n_samples_new, n_features)
            The resampled features.
            
        y_resampled : array-like, shape (n_samples_new,)
            The resampled target variable.
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Count samples in each class
        class_counts = Counter(y)
        
        # Find majority and minority classes
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [c for c in class_counts.keys() if c != majority_class]
        
        # If already balanced, return original data
        if len(minority_classes) == 0:
            return X, y
        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        for minority_class in minority_classes:
            # Calculate number of samples to generate for this minority class
            G = int((class_counts[majority_class] - class_counts[minority_class]) * self.beta)
            
            if G <= 0:
                continue
                
            # Get minority class samples
            X_minority = X[y == minority_class]
            
            # Find k nearest neighbors for each minority sample
            nn = NearestNeighbors(n_neighbors=self.k_neighbors+1)
            nn.fit(X)
            distances, indices = nn.kneighbors(X_minority)
            
            # Calculate density ratio
            r_i = np.zeros(len(X_minority))
            for i in range(len(X_minority)):
                minority_neighbors = sum(1 for idx in indices[i, 1:] if y[idx] != minority_class)
                r_i[i] = minority_neighbors / self.k_neighbors
                
            # Normalize r_i to create a distribution
            if r_i.sum() == 0:
                # If all r_i are 0, use uniform distribution
                r_i = np.ones(len(X_minority)) / len(X_minority)
            else:
                r_i = r_i / r_i.sum()
                
            # Calculate number of synthetic samples to generate for each minority instance
            n_samples_generate = np.rint(r_i * G).astype(int)
            
            # Generate synthetic samples
            for i, n_samples in enumerate(n_samples_generate):
                if n_samples == 0:
                    continue
                    
                # Get the k nearest neighbors for this minority sample
                nn_indices = indices[i, 1:]
                
                # Generate n_samples synthetic samples
                for _ in range(n_samples):
                    # Randomly select one of the k nearest neighbors
                    nn_idx = np.random.choice(nn_indices)
                    
                    # Generate a synthetic sample
                    dif = X[nn_idx] - X_minority[i]
                    gap = np.random.random()
                    synthetic_sample = X_minority[i] + gap * dif
                    
                    # Add the synthetic sample
                    X_resampled = np.vstack((X_resampled, synthetic_sample))
                    y_resampled = np.append(y_resampled, minority_class)
        
        return X_resampled, y_resampled

# --- Step 1: Load and Preprocess Dataset ---
print("Loading and preprocessing dataset...")
data = pd.read_csv("part-00001_preprocessed_dataset.csv")  # Replace with your dataset path

# Define features (X) and target (y)
target_col = "label"  # Adjust if needed
X = data.drop(columns=[target_col])  # Drop the target column
y = data[target_col]

# Encode categorical features (if any)
X = pd.get_dummies(X, drop_first=True)

# Encode target column if categorical
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# --- Step 2: Feature Selection ---
print("Selecting most informative features...")
# Select top 20 features based on mutual information
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_features = selector.get_support(indices=True)
print(f"Selected Features Indices: {selected_features}")

# --- Step 3: Split and Scale Dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Apply ADASYN for class balancing ---
print("Applying ADASYN to balance the training data...")
# Display class distribution before ADASYN
print("Class distribution before ADASYN:")
print(pd.Series(y_train).value_counts())

# Apply ADASYN to balance the dataset
adasyn = ADASYN(beta=1.0, k_neighbors=5, random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)

# Display class distribution after ADASYN
print("Class distribution after ADASYN:")
print(pd.Series(y_train_resampled).value_counts())
print(f"Original training set size: {X_train_scaled.shape[0]} samples")
print(f"Resampled training set size: {X_train_resampled.shape[0]} samples")

# --- Step 5: Define Weighted Naive Bayes ---
class WeightedNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.feature_weights = None

    def fit(self, X, y, feature_weights=None):
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        if feature_weights is None:
            feature_weights = np.ones(n_features)  # Default weights for features

        self.class_priors = np.bincount(y) / len(y)  # Estimate class priors
        self.feature_weights = feature_weights

        # Class-conditional probabilities (Gaussian assumption)
        self.class_means = np.zeros((n_classes, n_features))
        self.class_vars = np.zeros((n_classes, n_features))

        for c in range(n_classes):
            X_c = X[y == c]
            self.class_means[c, :] = np.mean(X_c, axis=0)
            self.class_vars[c, :] = np.var(X_c, axis=0)

    def predict(self, X):
        probs = self._calculate_log_likelihoods(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        probs = self._calculate_log_likelihoods(X)
        return np.exp(probs - np.max(probs, axis=1, keepdims=True))

    def _calculate_log_likelihoods(self, X):
        log_likelihoods = np.zeros((X.shape[0], len(self.class_priors)))

        for c in range(len(self.class_priors)):
            prior = np.log(self.class_priors[c])
            likelihood = -0.5 * (
                np.log(2 * np.pi * self.class_vars[c, :] + 1e-8) +
                ((X - self.class_means[c, :])**2) / (self.class_vars[c, :] + 1e-8)
            )
            # Weighted sum across features
            weighted_likelihood = (likelihood * self.feature_weights).sum(axis=1)
            log_likelihoods[:, c] = prior + weighted_likelihood

        return log_likelihoods

# --- Step 6: Optimize Feature Weights ---
def objective_function(weights, X, y):
    """
    Objective function to minimize weighted log loss.
    """
    wnb = WeightedNaiveBayes()
    wnb.fit(X, y, feature_weights=weights)
    y_pred_proba = wnb.predict_proba(X)
    log_loss_value = log_loss(y, y_pred_proba, labels=np.unique(y))
    print(f"Current Log Loss: {log_loss_value:.4f}")  # Monitor optimization progress
    return log_loss_value

# Initialize weights and bounds
initial_weights = np.ones(X_train_resampled.shape[1])
bounds = [(0.01, 10)] * X_train_resampled.shape[1]

print("Optimizing feature weights...")
result = minimize(
    fun=objective_function,
    x0=initial_weights,
    args=(X_train_resampled, y_train_resampled),
    method="L-BFGS-B",
    bounds=bounds,
    options={"disp": True, "maxiter": 50, "gtol": 1e-4}
)

optimized_weights = result.x
print("Optimized Feature Weights:", optimized_weights)

# --- Step 7: Train Final Weighted Naive Bayes Model ---
wnb = WeightedNaiveBayes()
wnb.fit(X_train_resampled, y_train_resampled, feature_weights=optimized_weights)

# --- Step 8: Evaluate Model ---
y_pred = wnb.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Weighted Naive Bayes Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
plt.tight_layout()
plt.show()

# --- Step 9: Analyze impact of ADASYN ---
# Train a model without ADASYN for comparison
print("\n--- Training model without ADASYN for comparison ---")
wnb_without_adasyn = WeightedNaiveBayes()
wnb_without_adasyn.fit(X_train_scaled, y_train, feature_weights=optimized_weights)

y_pred_without_adasyn = wnb_without_adasyn.predict(X_test_scaled)
accuracy_without_adasyn = accuracy_score(y_test, y_pred_without_adasyn)
print(f"Weighted Naive Bayes Accuracy (without ADASYN): {accuracy_without_adasyn:.4f}")

print("\nClassification Report (without ADASYN):")
print(classification_report(y_test, y_pred_without_adasyn))

# Compare results
print("\n--- Comparison of Results ---")
print(f"Accuracy with ADASYN: {accuracy:.4f}")
print(f"Accuracy without ADASYN: {accuracy_without_adasyn:.4f}")
print(f"Improvement: {(accuracy - accuracy_without_adasyn):.4f}")

# Plot class distribution before and after ADASYN
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(y=y_train)
plt.title("Class Distribution Before ADASYN")
plt.xlabel("Count")
plt.ylabel("Class")

plt.subplot(1, 2, 2)
sns.countplot(y=y_train_resampled)
plt.title("Class Distribution After ADASYN")
plt.xlabel("Count")
plt.ylabel("Class")

plt.tight_layout()
plt.show()