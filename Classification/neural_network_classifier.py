from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

# load data
data = load_breast_cancer()
print(data.target_names)
#print(data.feature_names)
print(data.target.shape)

# train, test split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
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
    layers.Dense(activation='sigmoid', units=1)])

# compile the model
# batch is a chunk of the training data, batch_size is a hyperparameter that
# defines the num of samples to go through before updating the internal model parameters
# epoch is the number of times model goes through all samples
# Batch Gradient Descent. Batch Size = Size of Training Set
# Stochastic Gradient Descent. Batch Size = 1
# Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# fit the model on train data
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=15, epochs = 100)

history_df = pd.DataFrame(history.history)
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))

# plot the loss and val loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['binary_accuracy'], label='binary_acc')
plt.plot(history.history['val_binary_accuracy'], label='val_binary_acc')
plt.legend()
plt.show()