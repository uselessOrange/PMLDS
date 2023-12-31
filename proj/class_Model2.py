from sklearn.model_selection import train_test_split

from GetData import getData


import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from GetData import getData

import matplotlib.pyplot as plt
class Model:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.history={}
        self.RMSE={}
        self.config=[]
        self.activationFinction = 'sigmoid'
        self.configs=[[2,1],[2,2,1]]
        self.num_iterations=10
        self.learning_rate = 0.1
        self.total_epochs = 50



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
        learning_rate = self.learning_rate
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse'])

        patience = int(np.round( self.total_epochs * 0.1))

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss',  # Monitor validation loss
          patience=patience,  # Stop training if no improvement for 5 consecutive epochs
          restore_best_weights=True  # Restore model weights to the best observed during training
          )
        return early_stopping_callback

    def train_Model(self,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):

        num_of_features = len(X1_train1.columns)

        early_stopping_callback = self.config_setup(num_of_features)
        history = self.model.fit(X1_train1, y1_train1, epochs=self.total_epochs, validation_data=(X1_validation, y1_validation), callbacks=[early_stopping_callback])
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

    configs=[]

    for i in range(n_of_configs):
        num_of_layers = np.random.randint(5, 100)
        level = np.random.randint(0, 500)
        num_of_layers = np.random.randint(5, 10)
        config = np.random.randint(10, 100,size=(num_of_layers))+level
        configs.append(list(config))
    return configs

def lossPlot(datasets,modelList,loss,val_loss):
    for i, dataset in enumerate(datasets):
        fig, axs = plt.subplots(1, len(modelList[0]), figsize=(30,5), sharey=True)
        for j, models in enumerate(modelList[i]):
            for k, model in enumerate(models):
                axs[j].plot(loss[i][j][k],'b')  # Plot loss for each permutation
                axs[j].plot(val_loss[i][j][k],'r')  # Plot val_loss for each permutation
            axs[j].set_title(f'Dataset {dataset}, Model {j+1}')
            axs[j].legend()
            plt.xlabel('epoch')
            plt.ylabel('loss')
        plt.tight_layout()
        plt.show()

def boxPlot(datasets,modelList,RMSE):
    # Plotting RMSE as Box Plots
    fig, axs = plt.subplots(len(datasets), 1, figsize=(10, 10),sharey=False)
    RMSE_RAW=[]
    RMSE_mean=[]
    for i, dataset in enumerate(datasets):
        data_for_boxplot = []  # Prepare data for boxplot
        RMSE_mean_dataset = []
        for j, models in enumerate(modelList[i]):
            RMSE_in_permutation = []
            for k, model in enumerate(models):
                RMSE_in_iteration=RMSE[i][j][k]['test']  # Append RMSE for each permutation and dataset
                RMSE_in_permutation.append(RMSE_in_iteration)
            RMSE_in_permutation_mean=np.mean(RMSE_in_permutation)
            data_for_boxplot.append(RMSE_in_permutation)
            RMSE_mean_dataset.append(RMSE_in_permutation_mean)
        axs[i].boxplot(data_for_boxplot)
        axs[i].set_title(f'RMSE for Dataset {dataset}')
        RMSE_RAW.append(data_for_boxplot)
        RMSE_mean.append(RMSE_mean_dataset)
    plt.tight_layout()
    plt.show()
    return RMSE_RAW,RMSE_mean

def plot_Configs(configs):
    fig, axs = plt.subplots(len(configs), figsize=(10, 2 * len(configs)),sharey=True,sharex=True)

    for i, config in enumerate(configs):
        axs[i].bar(np.arange(len(config)), config)
        axs[i].set_title(f'Configuration {i+1}')
        axs[i].set_xlabel('Layers')
        axs[i].set_ylabel('Number in Layer')
        axs[i].set_xticks(np.arange(len(config)))
        axs[i].set_xticklabels([f'Layer {j+1}' for j in range(len(config))])

    plt.tight_layout()
    plt.show()


def expDecay_of_learningRate(initial_learning_rate,final_learning_rate,total_epochs):
    # Calculate the decay factor
    decay_factor = final_learning_rate / initial_learning_rate
    # Calculate the decay rate per epoch
    decay_rate = decay_factor ** (1 / total_epochs)
    # Create a learning rate schedule with exponential decay
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_rate=decay_rate,
        decay_steps=1,
        staircase=True  # For discrete steps (staircase decay)
    )

    # Create an array of epochs
    epochs = np.arange(1, total_epochs + 1)

#    Calculate learning rates for each epoch using the exponential decay formula
    lr_values = initial_learning_rate * (decay_rate ** epochs)

    # Plotting the learning rate schedule
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, lr_values, label='Learning Rate', marker='o')
    plt.title('Exponential Decay Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    return lr_schedule


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
print(model.history)

model2=Model()

model2.load_Model('model_test.h5')
model2.get_RMSE(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)

print(model2.RMSE['test'])

"""




"""
dataFrame = getData()

X2=dataFrame['workableData']['FeatureDividedData']['X2']
y=dataFrame['workableData']['y']

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state=42)
X1_train1, X1_validation, y1_train1, y1_validation = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)

config = [2,2,1]

num_iterations = 2

model_iteration_set = Model()
model_iteration_set.config = config
modelList, loss, val_loss = model_iteration_set.Model_Across_Iterations(num_iterations,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)

model = modelList[0]

print(model.RMSE)
print(model.config)
print(model.history)

print(loss)
print(val_loss)
print(val_loss[0])
print(val_loss[0][0])
"""

"""
dataFrame = getData()
i=2
X2=dataFrame['workableData']['FeatureDividedData'][f'X{i}']
y=dataFrame['workableData']['y']

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state=42)
X1_train1, X1_validation, y1_train1, y1_validation = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)

config = [2,2,1]

num_iterations = 2

model_permutation_set = Model()
modelList, loss, val_loss = model_permutation_set.Model_Across_Permutations(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)

model = modelList[0][0]

print(model.RMSE)
print(model.config)
print(model.history)

print(loss)
print(val_loss)
print(val_loss[0])
print(val_loss[0][0])
print(val_loss[0][0][0])
"""
"""

datasets = [2,3]
n_of_configs = 3
configs = config_randomSearch(n_of_configs)
model_across_datasets = Model()
model_across_datasets.num_iterations = 4

model_across_datasets.configs=configs
modelList, loss, val_loss, RMSE = model_across_datasets.Model_Across_DataSets(datasets)
"""