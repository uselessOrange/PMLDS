from DataPipeline import dataPipeline
import pandas as pd
def getData():
    df_august = pd.read_csv('https://raw.githubusercontent.com/WitoldSurdej/PFML/master/apartments_pl_2023_08.csv')
    df_september = pd.read_csv('https://raw.githubusercontent.com/WitoldSurdej/PFML/master/apartments_pl_2023_09.csv')
    df_october = pd.read_csv('https://raw.githubusercontent.com/WitoldSurdej/PFML/master/apartments_pl_2023_10.csv')

    df_august['Month'] = 0
    df_september['Month'] = 1
    df_october['Month'] = 2

    frames = [df_august, df_september, df_october]
    df = pd.concat(frames)

    dataJourney = dataPipeline(df)

    return dataJourney