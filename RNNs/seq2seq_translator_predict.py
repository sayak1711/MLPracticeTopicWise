import json
import pickle
from keras.models import Model, load_model
from keras.layers import Input
import config as cfg
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


# obtained during training code execution
with open('outputs/model_configurations.pickle', 'rb') as mc:
    model_configurations = pickle.load(mc)
max_len_input, num_words, max_len_target, num_words_output = model_configurations

# load the model
trained_model = load_model("models/"+cfg.MODEL_NAME)
# get back the required layers
encoder_inputs = trained_model.input[0]
decoder_embedding = trained_model.get_layer(name='decoder_embedding')
decoder_lstm = trained_model.get_layer(name='decoder_lstm')
decoder_dense = trained_model.get_layer(name='decoder_dense')

_, state_h, state_c = trained_model.get_layer(name='encoder_lstm').output
encoder_states = [state_h, state_c]

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(cfg.LATENT_DIM,))
decoder_state_input_c = Input(shape=(cfg.LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs = Input(shape=(1, ))  # T=1 sequence
delo = decoder_embedding(decoder_inputs)  # decoder embedding layer output

decoder_outputs, state_h, state_c = decoder_lstm(
    delo, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

with open('outputs/idx2word_trans.json') as g:
    idx2word_trans = json.load(g)

with open('outputs/word2idx_outputs.json') as wi:
    word2idx_outputs = json.load(wi)


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
            word = idx2word_trans[str(idx)]
            output_sentence.append(word)

        # Update the decoder input
        # which is just the word just generated
        target_seq[0, 0] = idx

        # Update states
        states_value = [h, c]

    return ' '.join(output_sentence)


with open('outputs/tokenizer_input.pickle', 'rb') as t:
    tokenizer = pickle.load(t)

while True:
    input_sen = input('Enter sentence to be translated ')
    input_sequences = tokenizer.texts_to_sequences([input_sen])
    encoder_input_data = pad_sequences(input_sequences, maxlen=max_len_input)
    translation = decode_sequence(encoder_input_data)
    print(translation)
    ans = input("Continue? [Y/n]")
    if ans and ans.lower().strip() == 'n':
        break
