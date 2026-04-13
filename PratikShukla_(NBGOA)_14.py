import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

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

# --- Step 4: Define Weighted Naive Bayes ---
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

# --- Step 5: Optimize Feature Weights ---
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
initial_weights = np.ones(X_train_scaled.shape[1])
bounds = [(0.01, 10)] * X_train_scaled.shape[1]

print("Optimizing feature weights...")
result = minimize(
    fun=objective_function,
    x0=initial_weights,
    args=(X_train_scaled, y_train),
    method="L-BFGS-B",
    bounds=bounds,
    options={"disp": True, "maxiter": 50, "gtol": 1e-4}
)

optimized_weights = result.x
print("Optimized Feature Weights:", optimized_weights)

# --- Step 6: Train Final Weighted Naive Bayes Model ---
wnb = WeightedNaiveBayes()
wnb.fit(X_train_scaled, y_train, feature_weights=optimized_weights)

# --- Step 7: Evaluate Model ---
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
