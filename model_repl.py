import numpy as np
import operator
import sys

import trainer_core as core

def main():
    args = get_args()

    if not args:
        return

    (flip_model, colors_model, letters_model) = load_models(args)
    tokenizer = core.build_tokenizer(core.load_vocabulary())

    while True:
        text = input('> ').upper()
        if len(text) > 50:
            text = text[:50]

        flipped = run_model(flip_model, text, tokenizer)
        color = run_model(colors_model, text, tokenizer)
        letter = run_model(letters_model, text, tokenizer)

        print(str(translate_flip(flipped)))
        print(str(translate_colors(color)))
        print(str(translate_letters(letter)))

def get_args():
    args = {}

    try:
        assert len(sys.argv) == 4

        args['flips'] = sys.argv[1]
        args['colors'] = sys.argv[2]
        args['letters'] = sys.argv[3]

        return args
    except:
        print('Usage: model_repl.py <flips-model.h5> <colors-model.h5> <letters-model.h5>')

def load_models(args):
    colors = core.load_colors()
    letters = core.load_letters()

    vocab = core.load_vocabulary()
    vocab_size = len(vocab) + 1

    flip_model = core.build_model(vocab_size, 2)
    flip_model.load_weights(args['flips'])

    colors_model = core.build_model(vocab_size, len(colors) + 1)
    colors_model.load_weights(args['colors'])

    letters_model = core.build_model(vocab_size, len(letters) + 1)
    letters_model.load_weights(args['letters'])

    return (flip_model, colors_model, letters_model)

def run_model(model, text, tokenizer):
    tokenized = tokenizer.texts_to_sequences([text])
    padded = np.append(tokenized[0], np.zeros(core.INPUT_TEXT_LENGTH - len(tokenized[0])))
    padded = np.array(padded, ndmin=2)

    return model.predict(padded)[0]

def translate_flip(flip_cat):
    flips = ['Flip', 'Move']
    max_index, max_val = max_entry(flip_cat)
    return (flips[max_index], max_val)

def translate_colors(color_cat):
    colors = core.load_colors() + ['None']
    max_index, max_val = max_entry(color_cat)
    return (colors[max_index], max_val)

def translate_letters(letter_cat):
    letters = core.load_letters() + ['None']
    max_index, max_val = max_entry(letter_cat)
    return (letters[max_index], max_val)

def max_entry(categorization):
    return max(enumerate(categorization), key=operator.itemgetter(1))

if __name__ == '__main__':
    main()
