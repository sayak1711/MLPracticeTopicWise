# encoder--> decoder.
# encoder is bidirectional LSTM.
# between the 2 is attention mechanism.
# attention acts as a weight for each hidden state of encoder and thus passes the weighted sum to decoder.
# for each time step of the decoder.
# attention is calculated using a neural network where previous state of decoder and hidden state of encoder
# are inputs.
import configparser
import codecs
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, \
    Bidirectional, RepeatVector, Concatenate, Dot, Lambda
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt

try:
    import keras.backend as K

    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        from keras.layers import CuDNNLSTM as LSTM
        from keras.layers import CuDNNGRU as GRU
except:
    pass
glove_vectors = api.load('glove-wiki-gigaword-100')  # will download(if not already done before) and load
config = configparser.ConfigParser()
config.read('config.ini')
cfg = config['attention']


def softmax_over_time(x):
    assert K.ndim(x) > 2
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


input_texts = []  # sentence in original language
target_texts = []  # sentence in target language
target_texts_inputs = []  # sentence in target language offset by 1

# load in the data
# download the data at: http://www.manythings.org/anki/
t = 0
with codecs.open('../../datasets/spa-eng/spa.txt', encoding='utf-8', errors='ignore') as f:
    stuff = f.read()
lines = stuff.split('\n')
for line in lines:
    # only keep a limited number of samples
    t += 1
    if t > int(cfg['NUM_SAMPLES']):
        break

    # input and target are separated by tab
    if '\t' not in line:
        continue

    # split up the input and translation
    input_text, translation, _ = line.rstrip().split('\t')

    # make the target input and output
    # recall we'll be using teacher forcing
    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=int(cfg['MAX_NUM_WORDS']))
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))

# determine maximum length input sequence
max_len_input = max(len(s) for s in input_sequences)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=int(cfg['MAX_NUM_WORDS']), filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)  # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get the word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1  # size of output vocabulary

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)

# pad the sequences to make them same length using max_len
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("encoder_data.shape:", encoder_inputs.shape)
print("encoder_data[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_data[0]:", decoder_inputs[0])
print("decoder_data.shape:", decoder_inputs.shape)

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(int(cfg['MAX_NUM_WORDS']), len(word2idx_inputs) + 1)  # size of input vocabulary
embedding_matrix = np.zeros((num_words, int(cfg['EMBEDDING_DIM'])))
for word, i in word2idx_inputs.items():
    if i < int(cfg['MAX_NUM_WORDS']):
        if word in glove_vectors:
            embedding_matrix[i] = glove_vectors[word]

# create embedding layer
embedding_layer = Embedding(
    num_words,
    int(cfg['EMBEDDING_DIM']),
    weights=[embedding_matrix],
    input_length=max_len_input,
    # trainable=True
)

# one hot encoded decoder targets
decoder_targets_onehot = np.zeros(
    (
        len(target_texts),
        max_len_target,
        num_words_output,
    ),
    dtype='float32'
)

for i, dt in enumerate(decoder_targets):  # for each decoder target vector
    for t, word in enumerate(dt):  # for each word in max_len_target length decoder-target sequence
        if word > 0:  # if it is positive assign it 1 in one hot encoding
            decoder_targets_onehot[i, t, word] = 1

# Design the model
# ENCODER
encoder_inputs_placeholder = Input(shape=(max_len_input,))
encoder_input_el = embedding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(
    int(cfg['LATENT_DIM']),  # units/the dimensionality of the output space
    return_sequences=True
))
encoder_outputs = encoder(encoder_input_el)  # output hidden states from the encoder

# DECODER
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding_layer = Embedding(num_words_output, int(cfg['EMBEDDING_DIM']))
decoder_input_el = decoder_embedding_layer(decoder_inputs_placeholder)

# ATTENTION
st_1_repeater = RepeatVector(max_len_input)
concatenator = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')  # first layer of attention neural-net
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1)  # can also use Dot layer


def one_step_attention(h, st_1):  # creates one context-vector for one time step(t) of decoder part
    # we have h for all time-steps t' of encoder part
    # and the previous s (s(t-1)).
    # we concatenate s(t-1) and h(t') and pass it through a neural-net to get alpha(t').
    # now to be able to concatenate s(t-1) with h(which is multiple values for all t')
    # we need to repeat s(t-1) max_len_input times.
    st_1 = st_1_repeater(st_1)
    # now it has dimension (t'*LATENT_DIM_DECODER), t' is max_len_input.
    concatenated_stuff = concatenator([h, st_1])  # now dimension is t'*(LATENT_DIM_DEC+2*LATENT_DIM_ENC)
    x = attn_dense1(concatenated_stuff)
    alphas = attn_dense2(x)
    context_vector = attn_dot([alphas, h])
    return context_vector


decoder_lstm = LSTM(int(cfg['LATENT_DIM_DECODER']), return_state=True)
decoder_dense = Dense(num_words_output, activation='softmax')

initial_s = Input(shape=(int(cfg['LATENT_DIM_DECODER']),), name='s0')
initial_c = Input(shape=(int(cfg['LATENT_DIM_DECODER']),), name='c0')
context_and_last_word_concat_layer = Concatenate(axis=2)

s = initial_s
c = initial_c

outputs = []
for t in range(max_len_target):
    context = one_step_attention(encoder_outputs, s)
    selector = Lambda(lambda x: x[:, t:t + 1])
    xt = selector(decoder_input_el)

    decoder_lstm_input = context_and_last_word_concat_layer([context, xt])

    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)


def stack_and_transpose(x):
    # x is a list of length T, each element is a batch_size x output_vocab_size tensor
    x = K.stack(x)  # is now T x batch_size x output_vocab_size tensor
    x = K.permute_dimensions(x, pattern=(1, 0, 2))  # is now batch_size x T x output_vocab_size
    return x


# make it a layer
stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

# create the model
model = Model(
    inputs=[
      encoder_inputs_placeholder,
      decoder_inputs_placeholder,
      initial_s,
      initial_c,
    ],
    outputs=outputs
)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# train the model
z = np.zeros((len(encoder_inputs), int(cfg['LATENT_DIM_DECODER']))) # initial [s, c]
r = model.fit(
  [encoder_inputs, decoder_inputs, z, z], decoder_targets_onehot,
  batch_size=int(cfg['BATCH_SIZE']),
  epochs=int(cfg['EPOCHS']),
  validation_split=0.2
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

##### Make predictions #####
# As with the poetry example, we need to create another model
# that can take in the RNN state and previous word as input
# and accept a T=1 sequence.

# The encoder will be stand-alone
# From this we will get our initial decoder hidden state
# i.e. h(1), ..., h(Tx)
encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

# next we define a T=1 decoder model
encoder_outputs_as_input = Input(shape=(max_len_input, cfg['LATENT_DIM'] * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding_layer(decoder_inputs_single)

# no need to loop over attention steps this time because there is only one step
context = one_step_attention(encoder_outputs_as_input, initial_s)

# combine context with last word
decoder_lstm_input = context_and_last_word_concat_layer([context, decoder_inputs_single_x])

# lstm and final dense
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)

# note: we don't really need the final stack and tranpose
# because there's only 1 output
# it is already of size N x D
# no need to make it 1 x N x D --> N x 1 x D


# create the model object
decoder_model = Model(
    inputs=[
        decoder_inputs_single,
        encoder_outputs_as_input,
        initial_s,
        initial_c
    ],
    outputs=[decoder_outputs, s, c]
)

# map indexes back into real words
# so we can view the results
idx2word_eng = {v: k for k, v in word2idx_inputs.items()}
idx2word_trans = {v: k for k, v in word2idx_outputs.items()}


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    enc_out = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first character of target sequence with the start character.
    # NOTE: tokenizer lower-cases all words
    target_seq[0, 0] = word2idx_outputs['<sos>']

    # if we get this we break
    eos = word2idx_outputs['<eos>']

    # [s, c] will be updated in each loop iteration
    s = np.zeros((1, cfg['LATENT_DIM_DECODER']))
    c = np.zeros((1, cfg['LATENT_DIM_DECODER']))

    # Create the translation
    output_sentence = []
    for _ in range(max_len_target):
        o, s, c = decoder_model.predict([target_seq, enc_out, s, c])

        # Get next word
        idx = np.argmax(o.flatten())

        # End sentence of EOS
        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_trans[idx]
            output_sentence.append(word)

        # Update the decoder input
        # which is just the word just generated
        target_seq[0, 0] = idx

    return ' '.join(output_sentence)


while True:
    # Do some test translations
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i:i + 1]
    translation = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[i])
    print('Predicted translation:', translation)
    print('Actual translation:', target_texts[i])

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith('n'):
        break

