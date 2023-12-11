import numpy as np


import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self):
        self.model = []
        self.history={}
        self.RMSE={}
        self.config=[]
        self.activationFinction = 'relu'
        self.configs=[[2,1],[2,2,1]]
        self.num_iterations=10

    def get_RMSE(self):

        self.RMSE={'train':np.random.normal(1, 1000),'validation':np.random.normal(1, 1000),'test':np.random.normal(1, 1000)}



    def train_Model(self):

        size = np.random.randint(5, 15)
        loss = np.random.normal(1, 2, size=size)
        val_loss = np.random.normal(1, 2, size=size)
        self.history = {'loss': list(loss),'val_loss':list(val_loss)}

        self.get_RMSE()





    def Model_Across_Iterations(self):
        num_iterations = self.num_iterations
        modelList=[]
        loss = []
        val_loss = []
        RMSE = []
        for iteration in range(num_iterations):
            self.train_Model()
            loss.append(self.history['loss'])
            val_loss.append(self.history['val_loss'])
            modelList.append(self)
            RMSE.append(self.RMSE)
        return modelList, loss, val_loss, RMSE

    def Model_Across_Permutations(self):
        modelList = []
        loss = []
        val_loss = []
        RMSE = []
        configs = self.configs

        for config in configs:
            self.config = config
            modelList_per_iteration, loss_per_iteration, val_loss_per_iteration, RMSE_per_iteration = self.Model_Across_Iterations()
            modelList.append(modelList_per_iteration)
            loss.append(loss_per_iteration)
            val_loss.append(val_loss_per_iteration)
            RMSE.append(RMSE_per_iteration)
        return modelList, loss, val_loss, RMSE

    def Model_Across_DataSets(self,datasets):


      modelList = []
      loss = []
      val_loss = []
      RMSE = []  
      for i in datasets:




          modelList_per_permutation, loss_per_permutation, val_loss_per_permutation, RMSE_per_permutation = self.Model_Across_Permutations()

          modelList.append(modelList_per_permutation)
          loss.append(loss_per_permutation)
          val_loss.append(val_loss_per_permutation)
          RMSE.append(RMSE_per_permutation)
      return modelList, loss, val_loss, RMSE


def config_randomSearch(n_of_configs):
    num_of_layers = np.random.randint(5, 10)
    
    configs=[]

    for i in range(n_of_configs):
        level = np.random.randint(0, 20)
        num_of_layers = np.random.randint(5, 10)
        config = np.random.randint(10, 20,size=(num_of_layers))+level
        configs.append(list(config))
    return configs







"""

datasets = [2,3,5,7]
n_of_configs = 9
configs = config_randomSearch(n_of_configs)
model_across_datasets = Model()
model_across_datasets.num_iterations = 10

model_across_datasets.configs=configs
modelList, loss, val_loss, RMSE = model_across_datasets.Model_Across_DataSets(datasets)


model = modelList[0][0][0]

print(model.RMSE)
print(model.config)
print(model.history)

print(loss)
print(val_loss[0])
print(val_loss[0][0])
print(val_loss[0][0][0])
print(val_loss[0][0][0][0])
"""
"""
# Plotting Loss and Validation Loss
fig, axs = plt.subplots(len(datasets), len(modelList[0]), figsize=(15, 10), sharey=True)
for i, dataset in enumerate(datasets):
    for j, models in enumerate(modelList[i]):
        for k, model in enumerate(models):
            axs[i, j].plot(loss[i][j][k], label=f'Permutation {k+1}')  # Plot loss for each permutation
            axs[i, j].plot(val_loss[i][j][k], label=f'Val Loss Permutation {k+1}')  # Plot val_loss for each permutation
        axs[i, j].set_title(f'Dataset {dataset}, Model {j+1}')
        axs[i, j].legend()
plt.tight_layout()
plt.show()

"""
# Plotting Loss and Validation Loss
"""
for i, dataset in enumerate(datasets):
    rows = ceil(len(modelList[0])/2)
    columns = len(modelList[0])-rows
    fig, axs = plt.subplots(rows,columns, figsize=(10, 40), sharey=True)
    for j, models in enumerate(modelList[i]):
        for k, model in enumerate(models):
            axs[j].plot(loss[i][j][k], label=f'Permutation {k+1}')  # Plot loss for each permutation
            axs[j].plot(val_loss[i][j][k], label=f'Val Loss Permutation {k+1}')  # Plot val_loss for each permutation
        axs[j].set_title(f'Dataset {dataset}, Model {j+1}')
        axs[j].legend()
    plt.tight_layout()
    plt.show()
"""

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

#lossPlot(datasets,modelList,loss,val_loss)
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
#boxPlot(datasets,modelList,RMSE)


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

#plot_Configs(configs)