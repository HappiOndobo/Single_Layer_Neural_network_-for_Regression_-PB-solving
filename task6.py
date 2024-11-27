import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt


# Step 1: Data Preprocessing
# Load the Iris dataset
iris = load_iris()
data, target = iris.data, iris.target

# Normalize the data to range [0, 1]
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Convert to binary classification (Setosa vs. others)
binary_target = (target == 0).astype(int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    normalized_data, binary_target, test_size=0.2, random_state=42
)

# Display dataset details
print("First 5 rows of normalized training data:\n", X_train[:5])
print("\nFirst 5 rows of binary target labels:\n", y_train[:5])


# Step 2: Support Vector Machine Implementation
class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1 and 1

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)


# Step 3: Training the Model
# Initialize and train the SVM
svm = SupportVectorMachine(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

# Test the model
y_pred = svm.predict(X_test)

# Convert predictions to binary format (0 or 1)
y_pred_binary = np.where(y_pred > 0, 1, 0)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))
roc_auc = roc_auc_score(y_test, y_pred_binary)
print("\nROC-AUC Score:", roc_auc)


# Step 4: Visualization
# Plot the decision boundary for the first two features
def plot_svm(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn', edgecolors='k')

    # Plot hyperplane
    x0 = np.linspace(0, 1, 100)
    x1 = -(model.weights[0] * x0 + model.bias) / model.weights[1]
    plt.plot(x0, x1, label="Decision Boundary", color="blue")

    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1 (Normalized)")
    plt.ylabel("Feature 2 (Normalized)")
    plt.legend()
    plt.show()


# Plot decision boundary using first two features
plot_svm(X_test[:, :2], y_test, svm)
