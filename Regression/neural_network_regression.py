from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

X, y = load_boston(return_X_y=True)
print(X.shape)

# train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
N, D = X_train.shape # number of samples and num of dimensions

# scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build the model
model = keras.Sequential([
    layers.Dense(activation='relu', units=5, input_shape=[D]),
    layers.Dense(units=5, activation='relu'),
    layers.Dense(units=5, activation='relu'),
    layers.Dense(units=1)])

# compile the model
model.compile(optimizer='adam', loss='mae')

# fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# plot
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.show()