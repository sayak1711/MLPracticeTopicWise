import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

# load the data
# https://www.kaggle.com/uciml/sms-spam-collection-dataset
data = pd.read_csv('..\\datasets\\spam.csv', usecols=['v1', 'v2'], encoding='ISO-8859-1')
data = data.rename(columns={"v1": "label", "v2": "content"})
# convert ham to 0 and spam to 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print(data.head())

# split
df_train, df_test, y_train, y_test = train_test_split(data['content'], data['label'], test_size=0.33)

# convert texts to sequences
MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)  # sequence is a list of integer word indices
sequences_test = tokenizer.texts_to_sequences(df_test)

# word to integer mapping
word2idx = tokenizer.word_index
V = len(word2idx)
print(f'{V} unique tokens')

# we require N*T matrix..so padding is required
data_train = pad_sequences(sequences_train)
print(f'Shape of data_train tensor {data_train.shape}')
T = data_train.shape[1] # sequence length

data_test =  pad_sequences(sequences_test, maxlen=T) # greater than T will get truncated

D = 20 # embedding dimension hosen by us
M = 15 # number of units in hidden state

# build the model
model = Sequential([
    layers.Embedding(input_dim = V+1, output_dim=D, input_length=T),
    layers.LSTM(units=M, return_sequences=True),
    layers.GlobalMaxPooling1D(), # if we don't use return_sequences then we don't need this
    layers.Dense(units=1, activation='sigmoid')
])

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
history = model.fit(data_train, y_train, epochs=10, validation_data=(data_test, y_test))

# Plot loss per iteration
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show(block=False)
plt.pause(3)
plt.close()

# Plot accuracy per iteration
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show(block=False)
plt.pause(3)
plt.close()