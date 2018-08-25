#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:57:48 2018

@author: sheikhshahidurrahman
"""

import os
import numpy as np
import pandas as pd
from statistics import mode, median
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import clone_model, Sequential
from keras.layers import Flatten, Embedding, Dense, LSTM
from keras.utils.np_utils import to_categorical
from keras import regularizers


############################ Part 1 : Prediction of taster's identity ########################
############################ Load the data ############################
wineReviewFinal = pd.read_csv(filepath_or_buffer = "/Users/sheikhshahidurrahman/Downloads/wine-reviews/wineReviewFinal.csv")
wineReviewFinal.head()
wineReviewFinal.columns

## Separate response variable
wineReviewFinal.iloc[:,0].head()
x_train = wineReviewFinal['description']
type(x_train)
y_train_categories = pd.DataFrame(wineReviewFinal['taster_name'])
type(y_train_categories)

## Factorize the taster names
y_train, y_train_index = y_train_categories['taster_name'].factorize()
type(y_train)
len(set(y_train))
type(y_train_index)




############################ Building tensors of model data ########################
#### Input description ####

text = []
for description in x_train:
    text.append(description)
text[0]
len(text)

#### Tokenize the description ####
maxlen = [len(i) for i in text]
max(maxlen) # 829 is the maximum length of a description
np.mean(maxlen) # 244
mode(maxlen) # 239
median(maxlen) # 239
counter = Counter(maxlen)
type(counter)
sorted(counter.items(), key=itemgetter(0))

# So it seems we can pad the sequences till first 500 words in the description
maxDescription = 500
maxWords = 20000 # maximum no of most frequenct words, for our use case 20k is more than enough

tokenizer = Tokenizer(num_words = maxWords)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
type(sequences)
sequences[0]

word_index = tokenizer.word_index
len(word_index)
type(word_index)

data = pad_sequences(sequences, maxlen = maxDescription)
type(data)
data.shape
data[0]

# one-hot encoding for all the labels
y_train_one_hot = to_categorical(y_train)
type(y_train_one_hot)
y_train_one_hot.shape
y_train_one_hot[0]




############################ Model 1 : with self embedding and Dense signle layer ########################

## Model definition
model = Sequential()
model.add(Embedding(maxWords,32,input_length = maxDescription))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(19, activation = 'softmax'))
model.summary()

## K-fold cv with ratio maintained
skfolds = StratifiedKFold(n_splits = 10, random_state=1)

for train_index, test_index in skfolds.split(data, y_train):
    x_train_folds = data[train_index]
    y_train_folds = y_train_one_hot[train_index]
    x_val_folds = data[test_index]
    y_val_folds = y_train_one_hot[test_index]
    modelTemp = clone_model(model)
    modelTemp.compile(optimizer = 'rmsprop'
              , loss = 'categorical_crossentropy'
              , metrics = ['accuracy']
              )
    history = modelTemp.fit(x_train_folds
                            , y_train_folds
                            , epochs = 20
                            , batch_size = 512
                            , validation_data = (x_val_folds, y_val_folds))

## Plot the loss and accuracy to check for underfitting or overfitting
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo', label = "Training Loss")
plt.plot(epochs,val_loss,'b', label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs,acc,'bo', label = "Training Acc")
plt.plot(epochs,val_acc,'b', label = "Validation Acc")
plt.title("Training and Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
# max accuracy achieved 95% at 7th epoch

# Clearly not a case of over fitting till 7 epochs; after 7 epochs there's not 
# much of an improvement, leaving us computation resource waste

# Also as there is no prominent sign of the overfitting, we're under utilizing
# the network capacity


############################ Model 2 : self embedding(32) and multiple Dense layer ########################

## Model definition
model = Sequential()
model.add(Embedding(maxWords,32,input_length = maxDescription))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(19, activation = 'softmax'))
model.summary()

## K-fold cv with ratio maintained
skfolds = StratifiedKFold(n_splits = 10, random_state=1)

for train_index, test_index in skfolds.split(data, y_train):
    x_train_folds = data[train_index]
    y_train_folds = y_train_one_hot[train_index]
    x_val_folds = data[test_index]
    y_val_folds = y_train_one_hot[test_index]
    modelTemp = clone_model(model)
    modelTemp.compile(optimizer = 'rmsprop'
              , loss = 'categorical_crossentropy'
              , metrics = ['accuracy']
              )
    history = modelTemp.fit(x_train_folds
                            , y_train_folds
                            , epochs = 20
                            , batch_size = 512
                            , validation_data = (x_val_folds, y_val_folds))
    break

## Plot the loss and accuracy to check for underfitting or overfitting
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo', label = "Training Loss")
plt.plot(epochs,val_loss,'b', label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs,acc,'bo', label = "Training Acc")
plt.plot(epochs,val_acc,'b', label = "Validation Acc")
plt.title("Training and Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show() # max accuracy achieved 93.5% at 4th epoch

# Checked combination of higher embedding dimension and more dense layers
# Seems the network has reached its optimum performance
# The architecture is limited to certain accuracy level


############################ Model 3 : LSTM with self embedding ############################

## Model definition
model = Sequential()
model.add(Embedding(maxWords,32,input_length = maxDescription))
model.add(LSTM(100
               #, dropout=0.2
               #, recurrent_dropout=0.2
               , return_sequences=False)) # True when connecting to LSTM right next to it
# model.add(LSTM(32
#               , dropout=0.2
#               , recurrent_dropout=0.2
#               , return_sequences=False))
model.add(Dense(19, activation = 'softmax'))
model.summary()

## K-fold cv with ratio maintained
skfolds = StratifiedKFold(n_splits = 10, random_state=1)

for train_index, test_index in skfolds.split(data, y_train):
    x_train_folds = data[train_index]
    y_train_folds = y_train_one_hot[train_index]
    x_val_folds = data[test_index]
    y_val_folds = y_train_one_hot[test_index]
    modelTemp = clone_model(model)
    modelTemp.compile(optimizer = 'rmsprop'
              , loss = 'categorical_crossentropy'
              , metrics = ['accuracy']
              )
    history = modelTemp.fit(x_train_folds
                            , y_train_folds
                            , epochs = 20
                            , batch_size = 512
                            , validation_data = (x_val_folds, y_val_folds))
    break

## Plot the loss and accuracy to check for underfitting or overfitting
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo', label = "Training Loss")
plt.plot(epochs,val_loss,'b', label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs,acc,'bo', label = "Training Acc")
plt.plot(epochs,val_acc,'b', label = "Validation Acc")
plt.title("Training and Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show() # max accuracy achieved 95% at 15th epoch

# It seems like the chronological order is not much of an important
# factor to distinguish the writer of the review.

# It's more that the words representation make the difference between
# different writers. Hence the Dense network with Flattened representation
# works equally well as stacked LSTM layer.

# Two things worth trying 1. that we can use a pretrained word embedding
# instead of training a new one 2. Use this both with Dense and LSTM

# Also it's not much of a use to try and check the bi-directional LSTM
# network. So lets try only two more architectures, i.e. Embedding with 
# GloVe


############################ Model 4 : Dense with learned embedding ############################

## Get GloVe word embedding
glove_dir = '/Users/sheikhshahidurrahman/Downloads/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    #print(values[0])
    coefs = np.asarray(values[1:], dtype = 'float32')
    #print(values[1:])
    embeddings_index[word] = coefs
f.close()

embedding_dim = 100

embedding_matrix = np.zeros((maxWords, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < maxWords:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

## Model architecture
model = Sequential()
model.add(Embedding(maxWords,100,input_length = maxDescription))
model.add(Flatten())
model.add(Dense(64
               , kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
               , activation = 'relu'))
model.add(Dense(64
               , kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)
               , activation = 'relu'))
model.add(Dense(19, activation = 'softmax'))
model.summary()

# Freeze the embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

## K-fold cv with ratio maintained
skfolds = StratifiedKFold(n_splits = 10, random_state=1)

for train_index, test_index in skfolds.split(data, y_train):
    x_train_folds = data[train_index]
    y_train_folds = y_train_one_hot[train_index]
    x_val_folds = data[test_index]
    y_val_folds = y_train_one_hot[test_index]
    modelTemp = clone_model(model)
    modelTemp.compile(optimizer = 'rmsprop'
              , loss = 'categorical_crossentropy'
              , metrics = ['accuracy']
              )
    history = modelTemp.fit(x_train_folds
                            , y_train_folds
                            , epochs = 20
                            , batch_size = 512
                            , validation_data = (x_val_folds, y_val_folds))
    break

## Plot the loss and accuracy to check for underfitting or overfitting
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo', label = "Training Loss")
plt.plot(epochs,val_loss,'b', label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs,acc,'bo', label = "Training Acc")
plt.plot(epochs,val_acc,'b', label = "Validation Acc")
plt.title("Training and Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show() # max accuracy achieved 95% at 15th epoch


######################## Final Model : Dense with self embedding ########################

# We go ahead with the Dense network with self embedding having shown the best performance
# overall. Seems the word cloud is the best distinguisher of all the tasters.

maxDescription = 800
maxWords = 30000 # maximum no of most frequenct words, for our use case 20k is more than enough

tokenizer = Tokenizer(num_words = maxWords)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
type(sequences)
sequences[0]

word_index = tokenizer.word_index
len(word_index)
type(word_index)

data = pad_sequences(sequences, maxlen = maxDescription)
type(data)
data.shape
data[0]

# one-hot encoding for all the labels
y_train_one_hot = to_categorical(y_train)
type(y_train_one_hot)
y_train_one_hot.shape
y_train_one_hot[0]

# Model architecture
model = Sequential()
model.add(Embedding(maxWords,32,input_length = maxDescription))
model.add(Flatten())
model.add(Dense(64
                #, kernel_regularizer = regularizers.l1_l2(l1=0.001, l2=0.001)
                , activation = 'relu'))
model.add(Dense(19, activation = 'softmax'))
model.summary()

## K-fold cv with ratio maintained
skfolds = StratifiedKFold(n_splits = 10, random_state=1)

for train_index, test_index in skfolds.split(data, y_train):
    x_train_folds = data[train_index]
    y_train_folds = y_train_one_hot[train_index]
    x_val_folds = data[test_index]
    y_val_folds = y_train_one_hot[test_index]
    modelTemp = clone_model(model)
    modelTemp.compile(optimizer = 'rmsprop'
              , loss = 'categorical_crossentropy'
              , metrics = ['accuracy']
              )
    history = modelTemp.fit(x_train_folds
                            , y_train_folds
                            , epochs = 10
                            , batch_size = 512
                            , validation_data = (x_val_folds, y_val_folds))
    break

## Plot the loss and accuracy to check for underfitting or overfitting
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'bo', label = "Training Loss")
plt.plot(epochs,val_loss,'b', label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1,len(loss)+1)

plt.plot(epochs,acc,'bo', label = "Training Acc")
plt.plot(epochs,val_acc,'b', label = "Validation Acc")
plt.title("Training and Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()

# Overall accuracy before overfitting is ~96%

############################ End of Part 1 ########################




############################ Part 2 : Language Model for generating review per taster ########################

#### Data Preparation ####
wineReviewFinal = pd.read_csv(filepath_or_buffer = "/Users/sheikhshahidurrahman/Downloads/wine-reviews/wineReviewFinal.csv")
wineReviewFinal.head()
wineReviewFinal.columns

## Convert the names of the taster to categorical
wineReviewFinal.taster_name = pd.Categorical(wineReviewFinal.taster_name)
wineReviewFinal['tasterCode'] = wineReviewFinal.taster_name.cat.codes
#wineReviewFinal.taster_name.astype('category').cat.codes

## Create a mapping of name to index
nameMapping = wineReviewFinal.iloc[:,1:3]
nameMapping = nameMapping.groupby(['taster_name', 'tasterCode']).size().reset_index(name='Freq')
nameMapping = nameMapping.iloc[:,0:2]
type(nameMapping)
nameMapping = dict(zip(nameMapping.tasterCode, nameMapping.taster_name))
print(nameMapping)

## Filter dataset for taster identity
def createDataset(tasterCode):
    dataSet = wineReviewFinal.loc[wineReviewFinal['tasterCode'] == tasterCode]
    dataSet = dataSet['description']
    return dataSet

print(nameMapping) # Get taster name mapping, key:iteger, value:taster name
descriptionText = createDataset(12) # Change the index to get different taster name
len(descriptionText)

## Create list of the description for further processing
lines = list()
for line in descriptionText:
    lines.append(line)

## Tokenize the text
tokenizer = Tokenizer(num_words = 30000)
tokenizer.fit_on_texts(lines)
encoded = tokenizer.texts_to_sequences(lines)
len(encoded)
print(encoded[0:10])
assert(len(descriptionText)==len(encoded))

## Train test data preparation : Word sequences from one-all word as X: y as just the next word
# the reason for going this way is to avoid pre learned language model. The model sometimes doesn't
# fit to the use case as the corpus differes even after re-training on the new corpus.
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
sequences = list()
for l in encoded:
    for i in range(1, len(l)):
        sequence = l[:i+1]
        sequences.append(sequence)
        pass
    pass
len(sequences)
sequences[0:12]

# Pad the sequences
maxPad = max([len(i) for i in sequences])
sequences = pad_sequences(sequences, maxlen=maxPad, padding='pre')
sequences[0:12]
sequences.shape

# Split the data
X, y = sequences[:,:-1],sequences[:,-1]
# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)

#### Model for word prediction : Self embedding + LSTM ####
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=maxPad-1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy'
              , optimizer='adam' # rmsprop
              , metrics=['accuracy'])
model.summary()

## Fit the model
model.fit(X, y
          , epochs=50
          , batch_size = 512
          , verbose=1)

# Run the model to text prediction

def generate_seq(model, tokenizer, maxPad, seed_text, n_words):
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=maxPad-1, padding='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        in_text += ' ' + out_word
    return in_text

seed_text = "A barrel-aged surprise. The nose"
generatedText = generate_seq(model, tokenizer, maxPad, seed_text, 4)
print(generatedText)


############################ End of Part 2 ############################