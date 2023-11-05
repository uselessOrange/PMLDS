import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic data for binary classification
np.random.seed(0)
X = np.random.rand(100, 2)  # Two features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a shallow neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),  # Input layer with 2 features
    tf.keras.layers.Dense(units=4, activation='relu'),  # Shallow hidden layer with 4 neurons and ReLU activation
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Output layer for binary classification with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


# ... (your model training code) ...

# Save the trained model to a file
model.save("my_shallow_classifier_model.h5")
