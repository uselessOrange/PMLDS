from class_Model import Model
from sklearn.model_selection import train_test_split
from numpy import zeros
from GetData import getData

def Model_Across_Iterations(num_iterations,config,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test):
    modelList=zeros(num_iterations)
    for iteration in range(num_iterations):
        model=Model()
        model.config=config
        model.train_Model(X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)
        model.save_Model(f'model_{iteration}.h5')
        modelList[iteration]=model
    return modelList
'''
dataFrame = getData()

X2=dataFrame['workableData']['FeatureDividedData']['X2']
y=dataFrame['workableData']['y']

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.2, random_state=42)
X1_train1, X1_validation, y1_train1, y1_validation = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)

config = [2,2,1]

num_iterations = 2

Model_Across_Iterations(num_iterations,config,X1_train1, y1_train1,X1_validation, y1_validation,X_test,y_test)

'''