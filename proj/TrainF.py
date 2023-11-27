
import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt    



def trainModel():
# Define the neural network architecture
    model = tf.keras.Sequential()

    permutations = 3
    num_iterations=3

    model_struct=[]

    rmse_matrix=np.zeros([3,num_iterations,permutations])

    losses_per_permutation=[]
    val_losses_per_permutation=[]
    models=[]





    for permutation in range(permutations):


        losses=[]
        val_losses=[]


        model.add(Dense(2, activation='relu', input_shape=(2,)))
        #model.add(Dense(permutation+2, activation='relu'))
        model.add(Dense(1, activation='relu'))

        model_struct.append([permutation+1,1])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])





        for iteration in range(num_iterations):

            # Define a callback to collect training history
            history_callback = tf.keras.callbacks.History()

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=5,  # Stop training if no improvement for 5 consecutive epochs
            restore_best_weights=True  # Restore model weights to the best observed during training
            )



            history = model.fit(X1_train1, y1_train1, epochs=50, validation_data=(X1_validation, y1_validation), callbacks=[early_stopping_callback])

            models.append(model)

            losses.append(history.history['loss'])
            val_losses.append(history.history['val_loss'])



            y_predict_train = model.predict(X1_train1)
            y_predict_val = model.predict(X1_validation)
            y_predict_test = model.predict(X_test)




            rmse_matrix[0,iteration,permutation]=np.sqrt(mean_squared_error(y1_train1, y_predict_train))
            rmse_matrix[1,iteration,permutation]=np.sqrt(mean_squared_error(y1_validation, y_predict_val))
            rmse_matrix[2,iteration,permutation]=np.sqrt(mean_squared_error(y_test, y_predict_test))

        losses_per_permutation.append(losses)
        val_losses_per_permutation.append(val_losses)





    # Saving each model in the list to a directory
    for i, model in enumerate(models):
        model.save(f'/home/miko/anaconda3/envs/tf/git/PMLDS/proj/models/model_{i}.h5')  # Saving the model to disk with a unique filename
