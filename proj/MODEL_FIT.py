from tensorflow import tf
import numpy as np
from sklearn.metrics import mean_squared_error


def modelFit_testbench(all_histories_per_permutation,rmse_matrix,iteration):
    history = np.random(1,np.random(1))
    rmse_matrix[0,iteration]=np.random(1)
    rmse_matrix[1,iteration]=np.random(1)
    rmse_matrix[2,iteration]=np.random(1)



def modelFit(model,X1_train1,y1_train1,X1_validation,y1_validation,all_histories_per_permutation,X_test,y_test,rmse_matrix,iteration):

    history_callback = tf.keras.callbacks.History()

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop training if no improvement for 5 consecutive epochs
    restore_best_weights=True  # Restore model weights to the best observed during training
    )


    history = model.fit(X1_train1, y1_train1, epochs=50, validation_data=(X1_validation, y1_validation), callbacks=[early_stopping_callback])

    all_histories_per_permutation.append(history)




    y_predict_train = model.predict(X1_train1)
    y_predict_val = model.predict(X1_validation)
    y_predict_test = model.predict(X_test)



    rmse_matrix[0,iteration]=np.sqrt(mean_squared_error(y1_train1, y_predict_train))
    rmse_matrix[1,iteration]=np.sqrt(mean_squared_error(y1_validation, y_predict_val))
    rmse_matrix[2,iteration]=np.sqrt(mean_squared_error(y_test, y_predict_test))

    return rmse_matrix, all_histories_per_permutation


permutation = 1

num_iterations = 3
iteration=1
all_histories_per_permutation = []
rmse_matrix=np.zeros([3,num_iterations,permutation])

modelFit_testbench(all_histories_per_permutation,rmse_matrix,iteration)