import joblib
import numpy as np
# Load the saved model
loaded_model = joblib.load("shallow_neural_network_model.pkl")

#New data for prediction
new_data = np.array([[0.7, 0.8], [0.3, 0.2]])

# Make predictions on the new data
predictions = loaded_model.predict(new_data)

print("Predictions for new data:")
for data_point, prediction in zip(new_data, predictions):
    print(f"Input: {data_point}, Predicted Class: {prediction}")