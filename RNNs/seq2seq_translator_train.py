# ben-eng.zip downloaded from http://www.manythings.org/anki/
import json
# next 4 lines only required if running on GPU
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import gensim.downloader as api
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import codecs
import config as cfg
import pickle
glove_vectors = api.load('glove-wiki-gigaword-100') # will download(if not already done before) and load

# load the data
with codecs.open(cfg.TRAIN_DATA_PATH, encoding='utf-8', errors='ignore') as f:
    stuff = f.read()
all_lines = stuff.split('\n')
output_lines = []
input_lines = []
for line in all_lines:
    english, bengali, _ = line.split('	')
    output_lines.append(bengali)
    input_lines.append(english)

assert len(output_lines) == len(input_lines)

# process the data
# tokenize the inputs
tokenizer = Tokenizer(num_words=cfg.MAX_NUM_WORDS)
tokenizer.fit_on_texts(input_lines)
input_sequences = tokenizer.texts_to_sequences(input_lines)

# save tokenizer for use in prediction
with open('outputs/tokenizer_input.pickle', 'wb') as t:
    pickle.dump(tokenizer, t)

# get the word to index mapping for input language
word2idx_inputs = tokenizer.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)  # maximum length of input sequences

# pad the sequence
encoder_input_data = pad_sequences(input_sequences, maxlen=max_len_input)

# create the embedding matrix to be used in the embedding layer
num_words = min(cfg.MAX_NUM_WORDS, len(word2idx_inputs)+1)  # have to add 1 as it is 1 based indexing

embedding_matrix = np.zeros((num_words, cfg.EMBEDDING_DIM))
for word, i in word2idx_inputs.items():
    if i < cfg.MAX_NUM_WORDS:
        if word in glove_vectors:
            embedding_matrix[i] = glove_vectors[word]


# create the embedding layer
embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=cfg.EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input,

)

# prepare the decoder data
decoder_texts = []  # the target language texts
decoder_texts_inputs = []  # same but offset by 1 timestep to be used for teacher forcing
for output_line in output_lines:
    decoder_texts.append(output_line+' <eos>')
    decoder_texts_inputs.append('<sos> '+output_line)

tokenizer_o = Tokenizer(num_words=cfg.MAX_NUM_WORDS, filters='')
tokenizer_o.fit_on_texts(decoder_texts+decoder_texts_inputs)
decoder_target_sequences = tokenizer_o.texts_to_sequences(decoder_texts)  # this we have to one-hot later
decoder_input_sequences = tokenizer_o.texts_to_sequences(decoder_texts_inputs)

# get the word to index mapping for output language
word2idx_outputs = tokenizer_o.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))
# save idx2word and word2idx_outputs for prediction later
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}
with open('outputs/idx2word_trans.json', 'w') as iw:
    json.dump(idx2word_trans, iw)

with open('outputs/word2idx_outputs.json', 'w') as wi:
    json.dump(word2idx_outputs, wi)

num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in decoder_target_sequences)

# pad decoder data
decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=max_len_input, padding='post')  # maxlen same as enco
decoder_target_padded = pad_sequences(decoder_target_sequences, maxlen=max_len_target, padding='post')
# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
decoder_targets_one_hot = np.zeros(
  (
    len(input_lines),
    max_len_target,
    num_words_output
  ),
  dtype='float32'
)

for i, d in enumerate(decoder_target_padded):
    for t, word in enumerate(d):
        if word != 0:
            decoder_targets_one_hot[i, t, word] = 1

# design the encoder
encoder_inputs = Input(shape=(max_len_input, ), name='encoder_inputs')
eelo = embedding_layer(encoder_inputs)  # encoder embeddding layer output
encoder = LSTM(units=cfg.LATENT_DIM, return_state=True, name='encoder_lstm', dropout=0.5)
_, state_h, state_c = encoder(eelo)  # we don't need the output
encoder_states = [state_h, state_c]  # we just need hidden state and cell state for decoder

# design the decoder
decoder_inputs = Input(shape=(max_len_target, ))
decoder_embedding = Embedding(num_words_output, cfg.EMBEDDING_DIM, name='decoder_embedding')
delo = decoder_embedding(decoder_inputs)  # decoder embedding layer output
decoder_lstm = LSTM(units=cfg.LATENT_DIM, return_sequences=True, return_state=True, name='decoder_lstm', dropout=0.5)
decoder_outputs, _, _ = decoder_lstm(delo, initial_state=encoder_states)
decoder_dense = Dense(num_words_output, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit([encoder_input_data, decoder_input_data], decoder_targets_one_hot,
          batch_size=cfg.BATCH_SIZE,
          epochs=cfg.EPOCHS,
          validation_split=0.2)

# Save model
model.save('models/'+cfg.MODEL_NAME)

# save configurations
with open('outputs/model_configurations.pickle', 'wb') as mc:
    pickle.dump([max_len_input, num_words, max_len_target, num_words_output], mc)


# plot some data
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


# accuracies
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()



# INFERENCE (also coded as a seperate python file so as to seperate training and inference)
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(cfg.LATENT_DIM,))
decoder_state_input_c = Input(shape=(cfg.LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1, ))  # T=1 sequence
delo = decoder_embedding(decoder_inputs_single)  # decoder embedding layer output

decoder_outputs, dstate_h, dstate_c = decoder_lstm(
    delo, initial_state=decoder_states_inputs)

decoder_states = [dstate_h, dstate_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character.
    # NOTE: tokenizer lower-cases all words
    target_seq[0, 0] = word2idx_outputs['<sos>']

    # if we get this we break
    eos = word2idx_outputs['<eos>']

    # Create the translation
    output_sentence = []
    for _ in range(max_len_target):
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value
        )
        # Get next word
        idx = np.argmax(output_tokens[0, 0, :])

        # End sentence of EOS
        if eos == idx:
            break

        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)

        # Update the decoder input
        # which is just the word just generated
        target_seq[0, 0] = idx

        # Update states
        states_value = [h, c]

    return ' '.join(output_sentence)


while True:
    # Do some test translations
    i = np.random.choice(len(input_lines))
    input_seq = encoder_input_data[i:i+1]
    print(input_seq.shape)
    translation = decode_sequence(input_seq)
    print('-')
    print('Input:', input_lines[i])
    print('Translation:', translation)

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break
