import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib



# Generate synthetic data for binary classification
np.random.seed(0)
X = np.random.rand(100, 2)  # Two features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a shallow neural network using MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', solver='adam', max_iter=100)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')



# Save the trained model to a file
joblib.dump(model, "shallow_neural_network_model.pkl")