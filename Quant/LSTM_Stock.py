import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn import preprocessing

stock_name = '^GSPC'
seq_len = 22
d = 0.2
shape = [4, seq_len, 1]
neurons = [128, 128, 32, 1]
epochs = 300

def get_stock_data(stock_name, normalize=True):
    start = datetime.datetime(1950, 1, 1)
    end = datetime.date.today()
    df = web.DataReader(stock_name, "yahoo", start, end)
    df.drop(['Volume', 'Close'], 1, inplace=True)

    if normalize:
        min_max_scaler = preprocessing.MinMaxScalar()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
        return df

df = get_stock_data(stock_name, normalize=True)

def plot_stock(stock_name):
    df = get_stock_data(stock_name, normalize=True)
    print(df.head())
    plt.plot(df['Adj Close'], color='red', label='Adj Close')
    plt.legend(loc='best')
    plt.show()