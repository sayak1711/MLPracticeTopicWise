from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# load the data
data = load_wine(as_frame=True)
#print(data.target_names)
#print(data.feature_names)
X = data.data
y = data.target

# split data into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print(X_train)

# scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
N, D = X_train.shape
#print(X_train)
#print(y_train)

# build the model
# softmax for multiclass(also possible for 2 classes)
# sigmoid only supports 2 classes(binary)
model = keras.models.Sequential([
    layers.Dense(units=6, input_shape=[D], activation='relu'),
    layers.Dense(units=6, activation='relu'),
    layers.Dense(units=len(data.target_names), activation='softmax')
])

# compile the model
# for integer categories use sparse and for onehotencoded target use categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model on the train data
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# plot loss and eval_loss
plt.plot(history.history['loss'], label='loss') 
plt.plot(history.history['val_loss'], label='val_loss')
plt.show()

#plot accuracy and eval_accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.show()
