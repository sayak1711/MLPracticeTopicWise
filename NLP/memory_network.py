import os
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dense, Lambda, Reshape, Dot, Activation
import numpy as np
import keras.backend as k
from keras.models import Model
from keras.optimizers import RMSprop

dataset_path = '../../datasets/babi_tasks_1-20_v1-2/tasks_1-20_v1-2/en-10k/'
train_file_name = 'qa1_single-supporting-fact_train.txt'
test_file_name = 'qa1_single-supporting-fact_test.txt'


def get_data(file_name):
    with open(os.path.join(dataset_path, file_name)) as f:
        content = f.read()

    lines = content.split('\n')

    stories_and_qa = []
    cur_story = {}
    cur_qa = {}

    qa = False

    for i, line in enumerate(lines):
        line_no, stuff = line.split(' ', 1)
        stuff = stuff.split('\t')
        if len(stuff) == 1:  # story
            if qa:
                stories_and_qa.append({'story': cur_story, 'qa': cur_qa})
                cur_story = {}
                cur_qa = {}
            cur_story[i] = stuff[0]
            qa = False
        else:
            cur_qa[i] = stuff[:-1]
            qa = True

    return stories_and_qa


train_data = get_data(train_file_name)

test_data = get_data(test_file_name)

all_data = train_data + test_data
max_len_sentence_level = max(
    [len(line) for data in all_data for _, line in data['story'].items()])  # max len of sen in story among all stories
max_len_story_level = max([len(data['story']) for data in all_data])  # max num of line in a story
max_len_query_level = max([len(qa[0]) for data in all_data for _, qa in data['qa'].items()])  # max len of query for all queries

# create vocab
vocab = set()
for data in all_data:
    for _, line in data['story'].items():
        for word in nltk.word_tokenize(line):
            vocab.add(word)
    for _, qa in data['qa'].items():
        for item in qa:  # 2 items: query, answer
            for word in nltk.word_tokenize(item):
                vocab.add(word)

vocab = sorted(vocab)
vocab.insert(0, '<PAD>')
vocab_size = len(vocab)
print(len(vocab))
#print(vocab)
word2idx = {c: i for i, c in enumerate(vocab)}


def vectorize(data):
    # convert story, query, answer to number form by replacing word with integer
    story_vectors = []
    q_vectors = []
    a_vectors = []

    for di, d in enumerate(data):
        cur_story_line_vectors = []
        for _, line in d['story'].items():
            vector = [word2idx[word] for word in nltk.word_tokenize(line)]
            cur_story_line_vectors.append(vector)
        story_vectors.append(cur_story_line_vectors)

        for _, qa in d['qa'].items():
            vector = [word2idx[word] for word in nltk.word_tokenize(qa[0])]
            q_vectors.append(vector)
            vector = [word2idx[word] for word in nltk.word_tokenize(qa[1])]
            a_vectors.append(vector)

    story_vectors = [pad_sequences(line_vectors, maxlen=max_len_sentence_level) for line_vectors in story_vectors]
    q_vectors = pad_sequences(q_vectors, maxlen=max_len_query_level)
    a_vectors = np.array(a_vectors)
    return story_vectors, q_vectors, a_vectors


stories_train, queries_train, answers_train = vectorize(train_data)
stories_test, queries_test, answers_test = vectorize(test_data)

# now stack zero-filled lists to make all stories have same number of sentences (for this dataset they already are tho)
for i, st in enumerate(stories_train):
    stories_train[i] = np.concatenate([st, np.zeros((max_len_story_level-len(st), max_len_sentence_level), 'int')])
stories_train = np.stack(stories_train)  # since 0 is default value of axis, this converts the list into np.array

for i, st in enumerate(stories_test):
    stories_test[i] = np.concatenate([st, np.zeros((max_len_story_level-len(st), max_len_sentence_level), 'int')])
stories_test = np.stack(stories_test)

# create the model
embedding_dim = 15
input_story = Input((max_len_story_level, max_len_sentence_level))
embedded_story = Embedding(vocab_size, embedding_dim)(input_story)
#print(embedded_story.shape)
embedded_story = Lambda(lambda x: k.sum(x, axis=2))(embedded_story)  # add up the words to get one vector for sen
#print(input_story.shape, embedded_story.shape)  # see shapes before and after to understand what is happening

input_question_ = Input((max_len_query_level,))
embedded_question = Embedding(vocab_size, embedding_dim)(input_question_)
#print(embedded_question.shape)
# all word vectors in the single sentence query is added up to get one single vector for the query
embedded_question = Lambda(lambda x: k.sum(x, axis=1))(embedded_question)
#print(embedded_question.shape)

# reshape the embedded question such that it can be dotted with the story
print('To dot one vector with another along a same axis that axis needs to have the same value.')
embedded_question = Reshape((1, embedding_dim))(embedded_question)
#print(input_question_.shape, embedded_question.shape)

# now dot along the axis which is same size in story and question
dotted = Dot(axes=2)([embedded_story, embedded_question])
print(f'[embedded_story {embedded_story.shape}] dot [embedded_question {embedded_question.shape}] --> {dotted.shape}')
dotted = Reshape((max_len_story_level,))(dotted)  # flatten it
print(f'Shape of dot after flattening {dotted.shape}')
x = Activation('softmax')(dotted)
# now we get the weight for the story to understand how much weightage to be given to each sentence of the story
# with respect to the query
story_weights = Reshape((max_len_story_level, 1))(x)  # unflatten it to be dotted with the story later
print(f'Reshaped story_weights.shape {story_weights.shape} so as to dot it with embedded_story')

x = Dot(axes=1)([story_weights, embedded_story])
print(f'Shape after dotting story_weights with embedded_story {x.shape}')
x = Reshape((embedding_dim,))(x)
print(f'Reshaped to {x.shape} for passing through dense layer')

ans = Dense(vocab_size, activation='softmax')(x)
print(ans.shape)
model = Model([input_story, input_question_], ans)

# compile the model
model.compile(
    optimizer=RMSprop(lr=1e-2),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
r = model.fit(
  [stories_train, queries_train],
  answers_train,
  epochs=4,
  batch_size=32,
  validation_data=([stories_test, queries_test], answers_test)
)

# Check how we weight each input sentence given a story and question
debug_model = Model([input_story, input_question_], story_weights)

# choose a random story
story_idx = np.random.choice(len(train_data))

# get weights from debug model
i = stories_train[story_idx:story_idx+1]
q = queries_train[story_idx:story_idx+1]
w = debug_model.predict([i, q]).flatten()


story = [line for _, line in train_data[story_idx]['story'].items()]
question, ans = list(train_data[story_idx]['qa'].values())[0]
print("story:\n")
for i, line in enumerate(story):
    print("{:1.5f}".format(w[i]), "\t", line)

print("question:", question)
print("answer:", ans)

# TODO: 2 supporting fact


