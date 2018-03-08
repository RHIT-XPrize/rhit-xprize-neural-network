import numpy as np
import operator

try:
    from rhit_xprize_neural_network import trainer_core as core
except ImportError:
    import trainer_core as core

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
    if len(text) > 50:
        text = text[:50]

    tokenized = tokenizer.texts_to_sequences([text.upper()])
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
