import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("D:\\Work\\MLCodes\\Kaggle\\datasets\\IBM_03-01-2006_to_29-12-2017.csv", index_col='Date', parse_dates=['Date'])
#print(data.head())

# train set will be taken as before 2017 and test set from that for 'High' column
training_set = data[:'2016'].iloc[:,1:2].values # need it to be 2D
test_set = data['2017':].iloc[:,1:2].values
#print(training_set.shape)

# view it
data["High"][:'2016'].plot(figsize=(16,4),legend=True)
data["High"]['2017':].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])
plt.title('IBM stock price')
plt.show(block=False)
plt.pause(3)
plt.close()
print(test_set.shape)
# scale the data
sc = MinMaxScaler(feature_range=(0,1))
training_set = sc.fit_transform(training_set)

# create X, Y
def create_X_y(samples, T):
    X = []
    y = []
    for i in range(T, len(samples)):
        X.append(samples[i-T:i, 0])  # previous T data points
        y.append(samples[i, 0])
    return np.array(X).reshape((-1, T, 1)), np.array(y)

D = 1  # since only the high price
T = 60 # number of time steps
X_train, y_train = create_X_y(training_set, T)
dataset_total = pd.concat((data["High"][:'2016'],data["High"]['2017':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set)-T:].values # we need previous T too from train set
inputs = inputs.reshape(-1,1) # make it 2D
print(inputs.shape)
inputs  = sc.transform(inputs)
X_test, y_test = create_X_y(inputs, T)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# build the model
rnn_model = Sequential([
    layers.SimpleRNN(units=50, input_shape=(T, 1)),
    layers.Dense(units=1)
    ])
# compile
rnn_model.compile(optimizer='rmsprop', loss='mse')

# train
history = rnn_model.fit(X_train, y_train, epochs=20) #, validation_data=(X_test, y_test))

# plot loss
plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Losses SimpleRNN')
plt.show(block=False)
plt.pause(3)
plt.close()

plt.plot(rnn_model.predict(X_test), label='predicted')
plt.plot(y_test, label='actual')
plt.title('prediction comparison SimpleRNN')
plt.show(block=False)
plt.pause(3)
plt.close()

# now using LSTM
# build the model
lstm_model = Sequential([
    layers.LSTM(units=50, input_shape=(T, 1)),
    layers.Dense(units=1)
    ])
# compile
lstm_model.compile(optimizer='rmsprop', loss='mse')

# train
history = lstm_model.fit(X_train, y_train, epochs=20) #, validation_data=(X_test, y_test))

# plot loss
plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Losses LSTM')
plt.show(block=False)
plt.pause(3)
plt.close()

plt.plot(lstm_model.predict(X_test), label='predicted')
plt.plot(y_test, label='actual')
plt.title('prediction comparison LSTM')
plt.show(block=False)
plt.pause(3)
plt.close()

# now using GRU
gru_model = Sequential([
    layers.LSTM(units=50, input_shape=(T, 1)),
    layers.Dense(units=1)
    ])
# compile
gru_model.compile(optimizer='rmsprop', loss='mse')

# train
history = gru_model.fit(X_train, y_train, epochs=20) #, validation_data=(X_test, y_test))

# plot loss
plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Losses GRU')
plt.show(block=False)
plt.pause(3)
plt.close()

plt.plot(lstm_model.predict(X_test), label='predicted')
plt.plot(y_test, label='actual')
plt.title('prediction comparison GRU')

plt.show(block=False)
plt.pause(3)
plt.close()

