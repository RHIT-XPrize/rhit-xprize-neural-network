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

INPUT_TEXT_LENGTH = 50

def load_colors():
    with open('colors.txt') as colors_file:
        return list(map(lambda color: color.replace('\n', ''), colors_file.readlines()))

def load_letters():
    letter_nums = list(range(ord('A'), ord('H') + 1))
    return list(map(lambda i: chr(i), letter_nums))

def load_vocabulary():
    all_letters = list(map(chr, range(ord('A'), ord('Z') + 1)))
    return [' '] + all_letters

def build_tokenizer(vocabulary):
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(vocabulary)

    return tokenizer

def load_text(neural_in_file, tokenizer):
    text_df = pd.read_csv(neural_in_file, header=None, names=['Text'])

    text = tokenizer.texts_to_sequences(text_df.values[:, 0])

    for index, row in enumerate(text):
        text[index] = np.append(row, np.zeros(INPUT_TEXT_LENGTH - len(row)))

    return np.array(text)

def load_output(neural_out_file):
    output_df = pd.read_csv(neural_out_file, header=None)
    return np.array(output_df.values)

def build_model(vocab_size, output_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=INPUT_TEXT_LENGTH))
    model.add(LSTM(INPUT_TEXT_LENGTH))
    model.add(Dense(output_size, activation='softmax'))

    return model

def compile_model(model, loss='categorical_crossentropy'):
    model.compile(loss=loss,
                  optimizer='adam',
                  metrics=['accuracy'])

def train_model(model, text, output, model_file, iterations=10):
    for _ in range(iterations):
        model.fit(text,
                  output,
                  epochs=10,
                  batch_size=50,
                  verbose=True,
                  validation_split=0.20)

        model.save_weights(model_file)

def run_model(model, text, tokenizer):
    tokenized = tokenizer.texts_to_sequences([text])
    padded = np.append(tokenized[0], np.zeros(INPUT_TEXT_LENGTH - len(tokenized[0])))
    padded = np.array(padded, ndmin=2)

    return model.predict(padded)[0]

def translate_colors(color_cat):
    colors = load_colors()
    max_index, max_val = max_entry(color_cat)
    return (colors[max_index], max_val)

def translate_letters(letter_cat):
    letters = load_letters()
    max_index, max_val = max_entry(letter_cat)
    return (letters[max_index], max_val)

def max_entry(categorization):
    return max(enumerate(categorization), key=operator.itemgetter(1))
