import numpy as np
import sys

import trainer_core as core
import model_runner as runner

def main():
    args = get_args()

    if not args:
        return

    (flip_model, colors_model, letters_model) = runner.load_models(args)
    tokenizer = core.build_tokenizer(core.load_vocabulary())

    while True:
        text = input('> ').upper()
        if len(text) > 50:
            text = text[:50]

        flipped = runner.run_model(flip_model, text, tokenizer)
        color = runner.run_model(colors_model, text, tokenizer)
        letter = runner.run_model(letters_model, text, tokenizer)

        print(str(runner.translate_flip(flipped)))
        print(str(runner.translate_colors(color)))
        print(str(runner.translate_letters(letter)))

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

if __name__ == '__main__':
    main()
