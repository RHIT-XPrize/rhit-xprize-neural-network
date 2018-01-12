import pandas as pd
import numpy as np
import time
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dropout, Input
import tensorflow as tf

SYMBOLS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,')
N_SYMBOLS = len(SYMBOLS)
#might not need this? im not clear on embedding
TOKENS = dict((c, i) for i, c in enumerate(SYMBOLS))
MAX_LEN = 256
MAX_BLOCKS= 10;

# take in the instruction from the commander, i went with 100 max length just as a guess
words_input = Input(shape=(100,), name='words_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x1 = Embedding(output_dim=512, input_dim=10000, input_length=100)(words_input)
words_lstm=LSTM(32)(x1)

#the point to pass in
position_input = Input(shape=(16,), name='position_input')

#the state, max bblocks is set above
state_input = Input(shape=(MAX_BLOCKS,7),name='state_input')
state_input = keras.layers.Flatten()(state_input)
x = keras.layers.concatenate([words_lstm, position_input,state_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)

# # And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# model = Model(inputs=[words_lstm, position_input,state_input], outputs=[main_output, auxiliary_output])
# input_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy']);
