import pandas as pd
import numpy as np

def load_skab():
    data = pd.read_csv('datasets/raw/alldata_skab/alldata_skab.csv')

    data.index = pd.DatetimeIndex(data.datetime)
    data = data.drop(columns=['datetime'])

    data = data.dropna()

    labels = data.anomaly
    data = data.drop(columns=['anomaly', 'changepoint'])

    mask1 = data.index < '2020-02-09'
    mask2 = [a and b for a, b in zip(data.index > '2020-03-01', data.index < '2020-03-02')]
    mask3 = data.index > '2020-03-09'

    data1, labels1 = data[mask1], labels[mask1]
    data2, labels2 = data[mask2], labels[mask2]
    data3, labels3 = data[mask3], labels[mask3]

    return data1, labels1, data2, labels2, data3, labels3