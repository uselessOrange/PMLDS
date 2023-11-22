import tensorflow as tf
import numpy as np


def train_NN(X1_train1,y1_train1,X1_validation,y1_validation,model,num_iterations):

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    #accuracy_train=np.zeros([1,num_iterations])
    #accuracy_validation=np.zeros([1,num_iterations])
    #accuracy_test=np.zeros([1,num_iterations])

    #accuracy=np.zeros([3,num_iterations])

    for iteration in range(num_iterations):

        # Define a callback to collect training history
        history_callback = tf.keras.callbacks.History()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,  # Stop training if no improvement for 5 consecutive epochs
        restore_best_weights=True  # Restore model weights to the best observed during training
        )

        history = model.fit(X1_train1, y1_train1, epochs=50, validation_data=(X1_validation, y1_validation), callbacks=[early_stopping_callback])

    return history, model