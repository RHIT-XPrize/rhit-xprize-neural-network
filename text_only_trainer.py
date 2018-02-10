import pandas as pd
import numpy as np
import time
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Conv2D
from keras.models import Model, Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dropout, Input
import keras.preprocessing.text
import tensorflow as tf
import operator
import sys


if len(sys.argv) > 1:
    model_file = sys.argv[1]
else:
    model_file = 'textOnly.h5'

# # Load in Data
df_in = pd.read_csv('text-in.csv', header=None, names=['Text'])

all_colors = ['Red', 'Green', 'Blue', 'Orange', 'Yellow']
all_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

df_out = pd.read_csv('text-out.csv',
                     header=None,
                     names=(['Flip?'] + all_colors + all_letters))

# # Split and Format Data

# ## Initialize Tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)

letters = list(map(chr, range(ord('A'), ord('Z') + 1)))
tokenizer.fit_on_texts([' '] + letters)

VOCAB_SIZE = len(tokenizer.word_index) + 1


# ## Tokenize Data

total_entries = df_in.count()
training_entries = int(total_entries * 1.0)

train_words_in = df_in.iloc[:training_entries, :]
train_words_in = tokenizer.texts_to_sequences(train_words_in.values[:, 0])

for index, row in enumerate(train_words_in):
    train_words_in[index] = np.append(row, np.zeros(50 - len(row)))

train_words_in = np.array(train_words_in)

train_out = df_out.iloc[:training_entries, :].values

# # Build Model

# This model makeup (as well as the notable usage of `Tokenize`) comes from [here](https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/).

model = Sequential()
if len(sys.argv) > 2:
    model.load_weights(sys.argv[2])
else:
    model.add(Embedding(VOCAB_SIZE, 10, input_length=50))
    model.add(LSTM(50))
    model.add(Dense(df_out.shape[1], activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# # Train Model
for _ in range(10):
    model.fit(train_words_in,
              train_out,
              epochs=10,
              batch_size=50,
              verbose=True,
              validation_split=0.20)

    model.save_weights(model_file)


# # Manual Testing
def get_result_components(result_arr):
    return {
        'flip': result_arr[0],
        'colors': result_arr[1:len(all_colors) + 1],
        'letters': result_arr[len(all_colors) + 1:]
    }


def run_model(instruction):
    tokenized = tokenizer.texts_to_sequences([instruction])
    padded = np.append(tokenized[0], np.zeros(50 - len(tokenized[0])))
    padded = np.array(padded, ndmin=2)
    result = model.predict(padded)[0]

    return_data = {'raw': result}
    result = get_result_components(result)

    if result['flip'] > 0.5:
        return_data['flip?'] = True
    else:
        return_data['flip?'] = False

    max_index, max_val = max(enumerate(result['colors']), key=operator.itemgetter(1))
    return_data['color'] = (all_colors[max_index], max_val)


    max_index, max_val = max(enumerate(result['letters']), key=operator.itemgetter(1))
    return_data['letter'] = (all_letters[max_index], max_val)

    return return_data


result = run_model('MOVE THE RED C THERE')
print(str(result))
