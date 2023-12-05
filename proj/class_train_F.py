from sklearn.model_selection import train_test_split

from GetData import getData


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
        self.history={}
        self.RMSE={}
        self.config=[]
        self.activationFinction = 'relu'
        self.configs=[[2,1],[2,2,1]]
        self.num_iterations=10

    def get_RMSE(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):
        y_predict_train = self.model.predict(X1_train1)
        y_predict_val = self.model.predict(X1_validation)
        y_predict_test = self.model.predict(X_test)

        rmse_train=np.sqrt(mean_squared_error(y1_train1, y_predict_train))
        rmse_val=np.sqrt(mean_squared_error(y1_validation, y_predict_val))
        rmse_test=np.sqrt(mean_squared_error(y_test, y_predict_test))

        self.RMSE={'train':rmse_train,'validation':rmse_val,'test':rmse_test}

    def config_setup(self,num_of_features):
        

        #clearing the model to initiate new config
        del self.model
        self.model = tf.keras.Sequential()

        self.model.add(Input(shape=(num_of_features,)))
        activationfunction = self.activationFinction
     #   self.model.add(Input(shape=(self.config[0],)))
     #   self.model.add(Dense(self.config[1],activation='relu',input_shape=(self.config[0],)))
        for layer in range(len(self.config)):
            self.model.add(Dense(self.config[layer], activation=activationfunction))
        self.model.add(Dense(1))
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',  # Monitor validation loss
          patience=5,  # Stop training if no improvement for 5 consecutive epochs
          restore_best_weights=True  # Restore model weights to the best observed during training
          )
        return early_stopping_callback

    def train_Model(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):

        num_of_features = len(X1_train1.columns)

        early_stopping_callback = self.config_setup(num_of_features)
        history = self.model.fit(X1_train1, y1_train1, epochs=50, validation_data=(X1_validation, y1_validation), callbacks=[early_stopping_callback])
        self.history = {'loss': history.history['loss'],'val_loss':history.history['val_loss']}

        self.get_RMSE(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)

    def save_Model(self,name):
        self.model.save(name)
        #here insert code for saving history in a separate file. model.history is lost after saving model

    #def load_history(self):

    def load_Model(self,name):
        self.model=tf.keras.models.load_model(name)

    def Model_Across_Iterations(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):
        num_iterations = self.num_iterations
        modelList=[]
        loss = []
        val_loss = []
        RMSE = []
        for iteration in range(num_iterations):
            self.train_Model(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)
            loss.append(self.history['loss'])
            val_loss.append(self.history['val_loss'])
            #self.save_Model(f'model_{iteration}.h5')
            modelList.append(self)
            RMSE.append(self.RMSE)
        return modelList, loss, val_loss, RMSE

    def Model_Across_Permutations(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):
        modelList = []
        loss = []
        val_loss = []
        RMSE = []
        configs = self.configs

        for config in configs:
            self.config = config
            modelList_per_iteration, loss_per_iteration, val_loss_per_iteration, RMSE_per_iteration = self.Model_Across_Iterations(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)
            modelList.append(modelList_per_iteration)
            loss.append(loss_per_iteration)
            val_loss.append(val_loss_per_iteration)
            RMSE.append(RMSE_per_iteration)
        return modelList, loss, val_loss, RMSE

    def Model_Across_DataSets(self,datasets):
      dataFrame = getData()

      modelList = []
      loss = []
      val_loss = []
      RMSE = []

      for i in datasets:
          X2=dataFrame['workableData']['FeatureDividedData'][f'X{i}']
          y=dataFrame['workableData']['y']

          X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state=42)
          X1_train1, X1_validation, y1_train1, y1_validation = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)



          modelList_per_permutation, loss_per_permutation, val_loss_per_permutation, RMSE_per_permutation = self.Model_Across_Permutations(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)

          modelList.append(modelList_per_permutation)
          loss.append(loss_per_permutation)
          val_loss.append(val_loss_per_permutation)
          RMSE.append(RMSE_per_permutation)
      return modelList, loss, val_loss, RMSE


def config_randomSearch(n_of_configs):
    num_of_layers = np.random.randint(5, 10)

    configs=[]

    for i in range(n_of_configs):
        num_of_layers = np.random.randint(5, 10)
        config = np.random.randint(5, 15,size=(num_of_layers))
        configs.append(list(config))
    return configs


datasets = [2,3]
n_of_configs = 3
configs = config_randomSearch(n_of_configs)
model_across_datasets2 = Model()
model_across_datasets2.num_iterations = 4
model_across_datasets2.activationFinction = 'sigmoid'
model_across_datasets2.configs=configs
modelList2, loss2, val_loss2, RMSE2 = model_across_datasets2.Model_Across_DataSets(datasets)