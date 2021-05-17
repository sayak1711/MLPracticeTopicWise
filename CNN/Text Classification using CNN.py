# https://www.kaggle.com/au1206/text-classification-using-cnn/notebook
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.layers import Embedding, Input, Conv2D, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import gensim.downloader as api
glove_vectors = api.load('glove-wiki-gigaword-100')

TEXT_DATA_DIR = "..\\..\\datasets\\20_newsgroup\\20_newsgroup"
MAX_WORDS = 10000
MAX_SEQ_LENGTH = 1000
EMBEDDING_DIM = 100
num_filters = 512
# filter sizes of the different conv layers
filter_sizes = [3,4,5]
drop = 0.5
batch_size = 30
epochs = 25


texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
print(labels_index)

print('Found %s texts.' % len(texts))

# tokenize all texts
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word2idx = tokenizer.word_index
print("unique words : {}".format(len(word2idx)))

data = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)
labels = to_categorical(np.asarray(labels))
print(f'Data {data.shape}')
print(f'Label {labels.shape}')

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

num_words = min(MAX_WORDS, len(word2idx)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_WORDS:
        if word in glove_vectors:
            embedding_matrix[i] = glove_vectors[word]

embedding_layer = Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQ_LENGTH, trainable=False)
inputs = Input(shape=(MAX_SEQ_LENGTH,))
embedding = embedding_layer(inputs)
print(embedding.shape)
reshape = Reshape((MAX_SEQ_LENGTH, EMBEDDING_DIM, 1))(embedding)
print(reshape.shape)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
print(conv_0.shape)

maxpool_0 = MaxPool2D(pool_size=(MAX_SEQ_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQ_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_SEQ_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
print(maxpool_0.shape, maxpool_1.shape, maxpool_2.shape)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
print(concatenated_tensor.shape)
flatten = Flatten()(concatenated_tensor)
print(flatten.shape)
dropout = Dropout(drop)(flatten)
print(dropout.shape)
output = Dense(units=20, activation='softmax')(dropout)
print(output.shape)

model = Model(inputs=inputs, outputs=output)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("Traning Model...")
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()