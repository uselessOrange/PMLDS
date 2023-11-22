import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model("my_shallow_classifier_model.h5")

# Now you can use the loaded model for inference
predictions = loaded_model.predict(X_test)
