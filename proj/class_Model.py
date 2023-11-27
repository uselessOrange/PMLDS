import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from GetData import getData

class Model:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.hisory=[]
        self.RMSE={}
        self.config=[]



    def get_RMSE(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):
        y_predict_train = self.model.predict(X1_train1)
        y_predict_val = self.model.predict(X1_validation)
        y_predict_test = self.model.predict(X_test)




        rmse_train=np.sqrt(mean_squared_error(y1_train1, y_predict_train))
        rmse_val=np.sqrt(mean_squared_error(y1_validation, y_predict_val))
        rmse_test=np.sqrt(mean_squared_error(y_test, y_predict_test))

        self.RMSE={'train':rmse_train,'validation':rmse_val,'test':rmse_test}
    
    def config_setup(self):

        self.model.add(Input(shape=(self.config[0],)))
     #   self.model.add(Dense(self.config[1],activation='relu',input_shape=(self.config[0],)))
        for layer in range(len(self.config)-1):

            self.model.add(Dense(self.config[layer+1], activation='relu'))
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',  # Monitor validation loss
          patience=5,  # Stop training if no improvement for 5 consecutive epochs
          restore_best_weights=True  # Restore model weights to the best observed during training
          )
        return early_stopping_callback




    def train_Model(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):



        early_stopping_callback = self.config_setup()
        self.history = self.model.fit(X1_train1, y1_train1, epochs=50, validation_data=(X1_validation, y1_validation), callbacks=[early_stopping_callback])
        
        self.get_RMSE(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)
    
    def save_Model(self,name):
        self.model.save('name')
    
    def load_Model(self,name):
        self.model=tf.keras.models.load_model('name')
"""
dataFrame = getData()

X2=dataFrame['workableData']['FeatureDividedData']['X2']
y=dataFrame['workableData']['y']

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state=42)
X1_train1, X1_validation, y1_train1, y1_validation = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)

config = [2,2,1]

model=Model()
model.config=config
model.train_Model(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)
model.save_Model('model_test.h5')
print(model.RMSE['test'])

model2=Model()

model2.load_Model('model_test.h5')
model2.get_RMSE(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)
print(model2.RMSE['test'])
"""