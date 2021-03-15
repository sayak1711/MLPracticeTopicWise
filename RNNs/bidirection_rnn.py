import pandas as pd
import gensim.downloader as api
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np

MAX_SEQUENCE_LENGTH = 100
M=15
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5

glove_vectors = api.load('glove-wiki-gigaword-50') # will download(if not already done before) and load
print(glove_vectors.most_similar('twitter'))

#import inspect
#print(inspect.getsource(glove_vectors.__class__))

train_data = pd.read_csv('D:\\Work\\MLCodes\\datasets\\jigsaw-toxic-comment-classification-challenge\\train.csv')
sentences = train_data['comment_text']
#print(sentences.isnull().sum())
target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
targets = train_data[target_labels].values

# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)

# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    if word in glove_vectors:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = glove_vectors[word]


# load pre-trained word embeddings into an Embedding layer
# set trainable = False so as to keep the embeddings fixed
embedding_layer = layers.Embedding(
  input_dim=num_words,
  output_dim=EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)


# build the model
model = Sequential([
    embedding_layer,
    layers.Bidirectional(layers.LSTM(units=M, return_sequences=True)),
    layers.GlobalMaxPooling1D(),
    layers.Dense(units=len(target_labels), activation='sigmoid')
])

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
history = model.fit(data, targets, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))

# predict on test.csv data
test_data = pd.read_csv('D:\\Work\\MLCodes\\datasets\\jigsaw-toxic-comment-classification-challenge\\test.csv')
ids = pd.DataFrame(test_data.id)
test_sentences = test_data['comment_text']
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
p = model.predict(test_data)
df = pd.DataFrame(p, columns=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
df = pd.concat([ids, df], axis=1)
df.to_csv('jigsaw_toxic_results.csv', index=False)
