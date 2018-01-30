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

SYMBOLS = np.asarray(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,1234567890'))
N_SYMBOLS = len(SYMBOLS)
TOKENS = dict((c, i) for i, c in enumerate(SYMBOLS))
MAX_LEN = 256
MAX_BLOCKS= 10;

def tokenize_string(s):
    ret = np.zeros((MAX_LEN, N_SYMBOLS), dtype=bool)
    for i, char in enumerate(s):
        ret[i, TOKENS[char]] = 1
    return ret

def tokenize(a):
    return np.array(list(map(lambda s: tokenize_string(s), list(a))))

##function to build the net easily
def BuildNet(NODES,model):
    k = 1
    for nodes in NODES:
        model = Dense(nodes,activation='relu')(model)
        k = k+1
    return model
##replace later with better encoder. Encode colors
def color_cat(s):
    if s == 'GREEN':
        return 1
    elif s == 'BLUE':
        return 2
    elif s == 'RED':
        return 3
    else:
        return 4

## turn all the words into numbers, replace with more elgeant solution for varying amounts of letters and words later
def encode_state(state):
    state.block0_side1_letter = state.block0_side1_letter.apply(ord)-ord('A')+1
    state.block0_side2_letter = state.block0_side2_letter.apply(ord)-ord('A')+1
    state.block1_side1_letter = state.block1_side1_letter.apply(ord)-ord('A')+1
    state.block1_side2_letter = state.block1_side2_letter.apply(ord)-ord('A')+1
    state.block0_side1_color=state.block0_side1_color.apply(color_cat)
    state.block0_side2_color=state.block0_side2_color.apply(color_cat)
    state.block1_side1_color=state.block1_side1_color.apply(color_cat)
    state.block1_side2_color=state.block1_side2_color.apply(color_cat)
    return state;  
# take in the instruction from the commander, i went with 100 max length just as a guess
words_input = Input(shape=(MAX_LEN,N_SYMBOLS), name='words_input')

# # This embedding layer will encode the input sequence
# # into a sequence of dense 512-dimensional vectors.
# x1 = Embedding(output_dim=512, input_dim=10000, input_length=100)(words_input)
words_lstm=LSTM(32)(words_input)

#the point to pass in
position_input = Input(shape=(2,), name='position_input')

#the state, max bblocks is set above
state_input = Input(shape=(14,),name='state_input')
#state_input1 = keras.layers.Flatten()(state_input)
network_input = keras.layers.concatenate([words_lstm, position_input,state_input])

# # do actual building here, pull out later functionally do hyper parameter building
Nodes=[3,6,9]
network= BuildNet(Nodes,network_input);

block_id_output = Dense(1, activation='relu',name='block_id_output')(network)
point_output= Dense(2,activation='relu',name='point_output')(network)


final_network=Model(inputs=[state_input,words_input,position_input],outputs=[block_id_output,point_output])
final_network.compile(loss='mean_squared_error',
        optimizer='Adam',
        metrics=['accuracy']) 

##LOAD ALL THE DATA
df_in = pd.read_csv('C:/Users/bubulkr/rhit-xprize-neural-network/neural_in.csv')
df_out= pd.read_csv('C:/Users/bubulkr/rhit-xprize-neural-network/neural_out.csv')
train_points_in=df_in.iloc[:16000,15:17].values
train_words_in=df_in.iloc[:16000,14:15].values[:,0]
train_words_in=tokenize(train_words_in);
train_state_in=df_in.iloc[:16000,:14]
train_block_id_out=df_out.iloc[:16000,0].values
train_points_out = df_out.iloc[:16000,1:3].values
train_state_in = encode_state(train_state_in)
train_state_in = train_state_in.values
test_points_in=df_in.iloc[16000:,15:17].values
test_words_in=df_in.iloc[16000:,14:15].values[:,0]
test_words_in=tokenize(test_words_in);
test_state_in=df_in.iloc[16000:,:14]
test_block_id_out=df_out.iloc[16000:,0].values
test_points_out = df_out.iloc[16000:,1:3].values
test_state_in= encode_state(test_state_in)
test_state_in = test_state_in.values


##train
final_network.fit({'state_input':train_state_in, 'words_input':train_words_in, 'position_input':train_points_in},{'block_id_output':train_block_id_out,'point_output':train_points_out},epochs=10,batch_size=50)