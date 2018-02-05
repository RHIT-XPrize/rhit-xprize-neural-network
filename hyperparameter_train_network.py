import pandas as pd
import numpy as np
import time
import keras
from keras.layers import Input, Embedding, LSTM, Dense, Lambda
from keras.models import Model
from keras.layers.core import Dense,Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras.layers import LSTM, Dropout, Input
from keras import backend
import tensorflow as tf
import random

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
    return state
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
Nodes=[32, 65, 128, 32]
network = BuildNet(Nodes,network_input)

block_id_output = Dense(1, activation='relu',name='block_id_output')(network)
point_output= Dense(2,activation='relu',name='point_output')(network)

class NetworkConfig:
    MAX_BLOCK_LAYERS = 5
    MAX_POINT_LAYERS = 5
    MAX_BOTH_LAYERS = 5
    NETWORK_SIZES = [16, 32, 64, 128, 254, 512]
    def __init__(self, lstm_size, block_layers, point_layers, both_layers):
        self.lstm_size = lstm_size
        self.block_layers = block_layers
        self.point_layers = point_layers
        self.both_layers = both_layers

    @staticmethod
    def random_network_config():
        def create_layer_seq(num_layers):
            return list(map(lambda x: random.choice(NetworkConfig.NETWORK_SIZES),
                            [i for i in range(num_layers)]))
        num_block_layers = random.randint(1, NetworkConfig.MAX_BLOCK_LAYERS)
        num_point_layers = random.randint(1, NetworkConfig.MAX_BLOCK_LAYERS)
        num_both_layers = random.randint(1, NetworkConfig.MAX_BLOCK_LAYERS)

        lstm_size = random.choice(NetworkConfig.NETWORK_SIZES)
        block_layers = create_layer_seq(num_block_layers)
        point_layers = create_layer_seq(num_point_layers)
        both_layers = create_layer_seq(num_both_layers)

        return NetworkConfig(lstm_size, block_layers, point_layers, both_layers)

def create_network(MAX_LEN, N_SYMBOLS):
    config = NetworkConfig.random_network_config()

    words_input = Input(shape=(MAX_LEN,N_SYMBOLS), name='words_input')
    words_lstm=LSTM(config.lstm_size)(words_input)
    words_lstm = Dense(64, activation='relu')(words_lstm)

    position_input = Input(shape=(2,), name='position_input')

    network_input = keras.layers.concatenate([words_lstm, position_input,state_input])

    main_network = BuildNet(config.both_layers, network_input)
    main_output = Dense(1, activation='relu', name='main_output')(main_network)

    return Model(
        inputs=[state_input,words_input,position_input],
        outputs=[main_output]
    )

##LOAD ALL THE DATA
df_in = pd.read_csv('./neural_in.csv')
df_out= pd.read_csv('./neural_out.csv')

num_samples = df_in.shape[0]
num_samples_training = int(num_samples * 0.25)

train_points_in = df_in.iloc[:num_samples_training,15:17].values
train_words_in = df_in.iloc[:num_samples_training,14:15].values[:,0]
train_words_in = tokenize(train_words_in);
train_state_in = df_in.iloc[:num_samples_training,:14]
train_block_id_out = df_out.iloc[:num_samples_training,0].values
train_points_out = df_out.iloc[:num_samples_training,1:3].values
train_state_in = encode_state(train_state_in)
train_state_in = train_state_in.values

test_points_in = df_in.iloc[num_samples_training:,15:17].values
test_words_in = df_in.iloc[num_samples_training:,14:15].values[:,0]
test_words_in = tokenize(test_words_in);
test_state_in = df_in.iloc[num_samples_training:,:14]
test_block_id_out = df_out.iloc[num_samples_training:,0].values
test_points_out = df_out.iloc[num_samples_training:,1:3].values
test_state_in = encode_state(test_state_in)
test_state_in = test_state_in.values

train_main_out = df_out.iloc[:num_samples_training,0:1]
test_main_out = df_out.iloc[num_samples_training:,0:1]


np.array(train_state_in[:10])
np.array(train_words_in[:10])
np.array(train_points_in[:10])


##train
# final_network.fit({'state_input':train_state_in, 'words_input':train_words_in, 'position_input':train_points_in},{'block_id_output':train_block_id_out,'point_output':train_points_out},epochs=10,batch_size=50)

def main():
    epochs = 10
    num_iterations = 20

    best_result = None
    best_network = None
    for x in range(num_iterations):
        config = NetworkConfig.random_network_config()
        final_network = create_network(MAX_LEN, N_SYMBOLS)
        final_network.compile(
            loss='mean_squared_error',
            optimizer='Adam',
            metrics=['accuracy']
        )

        history = final_network.fit({
            'state_input': np.array(train_state_in),
            'words_input': np.array(train_words_in),
            'position_input': np.array(train_points_in)
        }, {
            'main_output': np.array(train_main_out)
        },
            epochs=epochs,
            batch_size=50,
            verbose=True
        )

        result = final_network.evaluate({
            'state_input': np.array(test_state_in),
            'words_input': np.array(test_words_in),
            'position_input': np.array(test_points_in)
        }, {
            'main_output': np.array(test_main_out)
        }, 
            verbose=True
        )

        if best_result is None or result[1] > best_result[1]:
            best_network = final_network
            best_result = result

    print('~~~~~ Done Training ~~~~~')
    print('Best result:', best_result)

    print('Spot test:')
    print(np.array(train_main_out[:10]))
    print(best_network.predict({
        'state_input': np.array(train_state_in[:10]),
        'words_input': np.array(train_words_in[:10]),
        'position_input': np.array(train_points_in[:10])
    }))


if __name__ == '__main__':
    main()

