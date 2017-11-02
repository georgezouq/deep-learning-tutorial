import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

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

'''
 Set last day Adjusted Close as y
'''
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1 # index starting from 0
    result = []

    for index  in range(len(data) - sequence_length): # maxmimum date = lastest data - sequence length
        result.append(data[index: index + sequence_length]) # index: index + 22days

    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split

    train = result[:index(row), :] # 90% date
    X_train = train[:, :-1][:, -1] # all data until data m
    y_train = train[:, -1][:, -1] # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_train, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]

X_train, y_train, X_test, y_test = load_data(df, seq_len)

X_train.shape[0], X_train.shape[1], X_train.shape[2]

y_train.shape[0]

'''
Build neural network
'''

def build_model2(layers, neurons, d):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], return_sequences=True)))
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], input_shape=(layers[1], return_sequences=False)))
    model.add(Dropout(d))

    model.add(Dense(neurons[2], kernel_initializer="uniform", activation="relu"))
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation="linear"))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

'''
Model Execution
'''

model = build_model2(shape, neurons, d)
model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=epochs,
    validation_split=0.1,
    verbose=1
)

'''
Result on training set and testing set
'''

def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

model_score(model, X_train, y_train, X_test, y_test)