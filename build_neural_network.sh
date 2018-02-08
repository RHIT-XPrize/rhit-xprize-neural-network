#!/bin/bash

if (( $# < 3))
then
    echo $#
    echo "Usage: $0 <num-games> <num-blocks> <model-h5-file> (-r)"
    exit 1
fi

DATA_FILE=$(mktemp)
NEURAL_IN_FILE=$(mktemp)
NEURAL_OUT_FILE=$(mktemp)
NUM_GAMES=$1
NUM_BLOCKS=$2
H5_FILE=$3

python3 generate_instructions.py $NUM_GAMES $NUM_BLOCKS $DATA_FILE
python3 split_file.py $NUM_BLOCKS $DATA_FILE $NEURAL_IN_FILE $NEURAL_OUT_FILE
python3 hyperparameter_train_network.py $NUM_BLOCKS $NEURAL_IN_FILE $NEURAL_OUT_FILE $H5_FILE

if (( $# > 3 ))
then
    rm -f $DATA_FILE $NEURAL_IN_FILE $NEURAL_OUT_FILE
else
    mv $DATA_FILE data.csv
    mv $NEURAL_IN_FILE neural_in.csv
    mv $NEURAL_OUT_FILE neural_out.csv
fi
