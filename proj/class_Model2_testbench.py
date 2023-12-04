from sklearn.model_selection import train_test_split

from GetData import getData


import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from GetData import getData

import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.history={}
        self.RMSE={}
        self.config=[]
        self.activationFinction = 'relu'
        self.configs=[[2,1],[2,2,1]]
        self.num_iterations=10

    def get_RMSE(self):

        self.RMSE={'train':np.random.normal(1, 1000),'validation':np.random.normal(1, 1000),'test':np.random.normal(1, 1000)}

    def config_setup(self,num_of_features):
        activationfunction = self.activationFinction

        #clearing the model to initiate new config
        del self.model
        self.model = tf.keras.Sequential()

        self.model.add(Input(shape=(num_of_features,)))

     #   self.model.add(Input(shape=(self.config[0],)))
     #   self.model.add(Dense(self.config[1],activation='relu',input_shape=(self.config[0],)))
        for layer in range(len(self.config)):
            self.model.add(Dense(self.config[layer], activation=activationfunction))

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',  # Monitor validation loss
          patience=5,  # Stop training if no improvement for 5 consecutive epochs
          restore_best_weights=True  # Restore model weights to the best observed during training
          )
        return early_stopping_callback

    def train_Model(self):

        #num_of_features = len(X1_train1.columns)

        #early_stopping_callback = self.config_setup(num_of_features)
        #history = self.model.fit(X1_train1, y1_train1, epochs=50, validation_data=(X1_validation, y1_validation), callbacks=[early_stopping_callback])
        size = np.random.randint(5, 15)
        loss = np.random.normal(1, 2, size=size)
        val_loss = np.random.normal(1, 2, size=size)
        self.history = {'loss': list(loss),'val_loss':list(val_loss)}

        self.get_RMSE()

    def save_Model(self,name):
        self.model.save(name)
        #here insert code for saving history in a separate file. model.history is lost after saving model

    #def load_history(self):

    def load_Model(self,name):
        self.model=tf.keras.models.load_model(name)

    def Model_Across_Iterations(self):
        num_iterations = self.num_iterations
        modelList=[]
        loss = []
        val_loss = []
        for iteration in range(num_iterations):
            self.train_Model()
            loss.append(self.history['loss'])
            val_loss.append(self.history['val_loss'])
            #self.save_Model(f'model_{iteration}.h5')
            modelList.append(self)
        return modelList, loss, val_loss

    def Model_Across_Permutations(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):
        modelList = []
        loss = []
        val_loss = []
        configs = self.configs

        for config in configs:
            self.config = config
            modelList_per_iteration, loss_per_iteration, val_loss_per_iteration = self.Model_Across_Iterations()
            modelList.append(modelList_per_iteration)
            loss.append(loss_per_iteration)
            val_loss.append(val_loss_per_iteration)
        return modelList, loss, val_loss

    def Model_Across_DataSets(self,datasets):
      dataFrame = getData()

      modelList = []
      loss = []
      val_loss = []

      for i in datasets:
          X2=dataFrame['workableData']['FeatureDividedData'][f'X{i}']
          y=dataFrame['workableData']['y']

          X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state=42)
          X1_train1, X1_validation, y1_train1, y1_validation = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)



          modelList_per_permutation, loss_per_permutation, val_loss_per_permutation = self.Model_Across_Permutations(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)

          modelList.append(modelList_per_permutation)
          loss.append(loss_per_permutation)
          val_loss.append(val_loss_per_permutation)
      return modelList, loss, val_loss




datasets = [2,3]

model_across_datasets = Model()
model_across_datasets.num_iterations = 1
modelList, loss, val_loss = model_across_datasets.Model_Across_DataSets(datasets)


model = modelList[0][0][0]

print(model.RMSE)
print(model.config)
print(model.history)

print(loss)
print(val_loss[0])
print(val_loss[0][0])
print(val_loss[0][0][0])
print(val_loss[0][0][0][0])