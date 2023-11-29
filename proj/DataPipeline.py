import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#example
"""
df_august = pd.read_csv('https://raw.githubusercontent.com/WitoldSurdej/PFML/master/apartments_pl_2023_08.csv')
df_september = pd.read_csv('https://raw.githubusercontent.com/WitoldSurdej/PFML/master/apartments_pl_2023_09.csv')
df_october = pd.read_csv('https://raw.githubusercontent.com/WitoldSurdej/PFML/master/apartments_pl_2023_10.csv')

df_august['Month'] = 0
df_september['Month'] = 1
df_october['Month'] = 2

frames = [df_august, df_september, df_october]
df = pd.concat(frames)


"""
def dataPipeline(df):


    
    df_live, df_backup = train_test_split(df,test_size=0.3, random_state=42)
    
    num_cols = df_live.select_dtypes([np.number]).columns
    df_nums = df_live[num_cols].reset_index(drop=True)
    
    X = df_nums.loc[:,df_nums.columns != 'price']
    y = df_nums['price'].values
    
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)
    
    selected_columns_2 = ['squareMeters', 'longitude']
    selected_columns_3 = ['squareMeters', 'longitude', 'poiCount']
    selected_columns_5 = ['squareMeters', 'longitude', 'poiCount', 'rooms', 'centreDistance']
    selected_columns_7 = ['squareMeters', 'longitude', 'poiCount', 'rooms', 'centreDistance', 'clinicDistance', 'kindergartenDistance']
    
    X2=X_normalized_df[selected_columns_2]
    X3=X_normalized_df[selected_columns_3]
    X5=X_normalized_df[selected_columns_5]
    X7=X_normalized_df[selected_columns_7]

    FeatureDividedData = {'X2':X2,'X3':X3,'X5':X5,'X7':X7}

    workableData = {'X':X,'y':y,'X_normalized':X_normalized,'X_normalized_df':X_normalized_df,'FeatureDividedData':FeatureDividedData}

    dataJourney={'df_live':df_live,'df_backup':df_backup,'df_nums':df_nums,'scaler':scaler,'workableData':workableData}

    return dataJourney

#dataJourney = dataPipeline(df)




