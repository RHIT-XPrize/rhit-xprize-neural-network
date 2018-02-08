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
import sys

SYMBOLS = np.asarray(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,1234567890'))
N_SYMBOLS = len(SYMBOLS)
TOKENS = dict((c, i) for i, c in enumerate(SYMBOLS))
MAX_LEN = 50

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
    raise RuntimeError('Unrecognized color:', s)

## turn all the words into numbers, replace with more elgeant solution for varying amounts of letters and words later
def encode_states(states):
    for curr_state in states:
        assert len(curr_state) % 7 == 0
        for block_offset in range(0, len(curr_state), 7):
            curr_state[block_offset + 1] = \
                ord(curr_state[block_offset + 1]) - ord('A') + 1
            curr_state[block_offset + 3] = \
                ord(curr_state[block_offset + 3]) - ord('A') + 1
            curr_state[block_offset + 2] = \
                color_cat(curr_state[block_offset + 2])
            curr_state[block_offset + 4] = \
                color_cat(curr_state[block_offset + 4])

    return states

class NetworkConfig:
    MAX_BLOCK_LAYERS = 5
    MAX_POINT_LAYERS = 5
    MAX_BOTH_LAYERS = 5
    NETWORK_SIZES = [16, 32, 64, 128, 254, 512]
    def __init__(self, num_blocks, lstm_size, block_layers, point_layers, both_layers):
        self.lstm_size = lstm_size
        self.block_layers = block_layers
        self.point_layers = point_layers
        self.both_layers = both_layers
        self.num_blocks = num_blocks

    @staticmethod
    def random_network_config(num_blocks):
        def create_layer_seq(num_layers):
            return list(map(lambda x: random.choice(NetworkConfig.NETWORK_SIZES),
                            list(range(num_layers))))
        num_block_layers = random.randint(1, NetworkConfig.MAX_BLOCK_LAYERS)
        num_point_layers = random.randint(1, NetworkConfig.MAX_BLOCK_LAYERS)
        num_both_layers = random.randint(1, NetworkConfig.MAX_BLOCK_LAYERS)

        lstm_size = random.choice(NetworkConfig.NETWORK_SIZES)
        block_layers = create_layer_seq(num_block_layers)
        point_layers = create_layer_seq(num_point_layers)
        both_layers = create_layer_seq(num_both_layers)

        return NetworkConfig(num_blocks, lstm_size, block_layers, point_layers, both_layers)

def create_network(num_blocks, MAX_LEN, N_SYMBOLS):
    config = NetworkConfig.random_network_config(num_blocks)

    state_input = Input(shape=(7*num_blocks,),name='state_input')
    words_input = Input(shape=(MAX_LEN,N_SYMBOLS), name='words_input')
    words_lstm=LSTM(config.lstm_size)(words_input)
    words_lstm = Dense(128, activation='relu')(words_lstm)

    network_input = keras.layers.concatenate([words_lstm, state_input])

    main_network = BuildNet(config.both_layers, network_input)
    main_output = Dense(config.num_blocks, activation='softmax', name='main_output')(main_network)

    return Model(
        inputs=[state_input,words_input],
        outputs=[main_output]
    )

def load_data(neural_in, neural_out):
    df_in = pd.read_csv(neural_in, header=None)
    df_out= pd.read_csv(neural_out, header=None)

    num_samples = df_in.shape[0]
    num_samples_training = int(num_samples * 0.25)

    train_words_in = df_in.iloc[:num_samples_training,-1].values

    train_words_in = tokenize(train_words_in)
    train_state_in = df_in.iloc[:num_samples_training,:-1]
    train_state_in = encode_states(train_state_in.values)

    test_words_in = df_in.iloc[num_samples_training:,-1].values
    test_words_in = tokenize(test_words_in);
    test_state_in = df_in.iloc[num_samples_training:,:-1]
    test_state_in = encode_states(test_state_in.values)

    train_main_out = df_out.iloc[:num_samples_training,]
    test_main_out = df_out.iloc[num_samples_training:,]

    return {
        'train_words_in': np.array(train_words_in),
        'train_state_in': np.array(train_state_in),
        'train_main_out': np.array(train_main_out),
        'test_words_in': np.array(test_words_in),
        'test_state_in': np.array(test_state_in),
        'test_main_out': np.array(test_main_out)
    }

def load_args():
    if len(sys.argv) != 5:
        print('Usage: hyperparameter_train_network.py <num-blocks> <neural-in-file> <neural-out-file> <h5-out-file>')
        return None

    args = {}

    try:
        args['num-blocks'] = int(sys.argv[1])
    except:
        print('Number of blocks must be an integer')

    args['neural-in-file'] = sys.argv[2]
    args['neural-out-file'] = sys.argv[3]
    args['h5-out-file'] = sys.argv[4]

    return args

def main():
    args = load_args()
    if not args:
        print('Aborting')
        return

    epochs = 5
    num_iterations = 2
    num_blocks = args['num-blocks']

    data = load_data(args['neural-in-file'], args['neural-out-file'])

    best_result = []
    best_network = None
    for x in range(num_iterations):
        config = NetworkConfig.random_network_config(num_blocks)
        final_network = create_network(num_blocks, MAX_LEN, N_SYMBOLS)
        final_network.compile(
            loss='mean_squared_error',
            optimizer='Adam',
            metrics=['accuracy']
        )

        history = final_network.fit({
            'state_input': data['train_state_in'],
            'words_input': data['train_words_in']
        }, {
            'main_output': data['train_main_out']
        },
            epochs=epochs,
            batch_size=50,
            verbose=True
        )

        result = final_network.evaluate({
            'state_input': data['test_state_in'],
            'words_input': data['test_words_in']
        }, {
            'main_output': data['test_main_out']
        },
            verbose=True
        )

        if best_result == [] or result[1] > best_result[1]:
            best_network = final_network
            best_result = result

    best_network.save(args['h5-out-file'])

    print('~~~~~ Done Training ~~~~~')
    print('Best result:', best_result)

    print('Spot test:')
    print(np.array(data['train_main_out'][:10]))
    print(best_network.predict({
        'state_input': data['train_state_in'][:10],
        'words_input': data['train_words_in'][:10]
    }))


if __name__ == '__main__':
    main()
