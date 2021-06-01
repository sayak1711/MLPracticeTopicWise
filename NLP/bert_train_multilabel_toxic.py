# -*- coding: utf-8 -*-
"""bert-train-multilabel-toxic.ipynb
"""

import os
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

NUM_WORDS = 50000
MAXLEN = 150
DATA_DIR = "data"

'''
!ls ../input/jigsaw-toxic-comment-classification-challenge
!mkdir data
!unzip ../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip -d data/
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip -d data/
!unzip ../input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip -d data/
!ls
'''

train_df = pd.read_csv(os.path.join(DATA_DIR,'train.csv'))
texts = train_df['comment_text'].values
target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']
targets = train_df[target_labels].values
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, targets, test_size=.2)

print(train_texts.shape)
print(train_labels.shape)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")

print(train_encodings['input_ids'].shape)

val_encodings = tokenizer(list(val_texts), padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")

# attention_mask tells the model to not attend to the 0s which have been added as padding just to
# make all tensors same length. token_type_ids seems to be the sentence id indicator i.e in position of all tokens
# belonging to sentence 0 will be a 0 and so on
print(train_encodings)

# we have to convert our encodings into torch.utils.data.Dataset for model to be able to train
class ModelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # .clone().detach()
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ModelDataset(train_encodings, train_labels)
val_dataset = ModelDataset(val_encodings, val_labels)
#train_dataset.encodings = train_dataset.encodings.to(device)
#val_dataset.encodings = val_dataset.encodings.to(device)

print(train_dataset.labels.shape)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(target_labels))
#model.to(device)

'''
!mkdir model
!mkdir model/results
!mkdir model/logs
'''

training_args = TrainingArguments(
    output_dir='model/results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='model/logs',            # directory for storing logs
    logging_steps=10,
)

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

trainer = MultilabelTrainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset
)

trainer.train()
