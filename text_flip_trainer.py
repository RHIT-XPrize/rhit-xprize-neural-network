import sys

import trainer_core as core

def main():
    args = get_args()
    if not args:
        return

    vocabulary = core.load_vocabulary()
    vocabulary_size = len(vocabulary) + 1

    tokenizer = core.build_tokenizer(vocabulary)

    text = core.load_text(args['text-file'], tokenizer)
    flipped = core.load_output(args['flip-file'])

    model = core.build_model(vocabulary_size, 1)
    if 'model-input' in args:
        model.load_weights(args['model-input'])

    core.compile_model(model, True)

    core.train_model(model, text, flipped, args['model-output'])

def get_args():
    try:
        args = {}

        assert len(sys.argv) >= 4

        args['text-file'] = sys.argv[1]
        args['flip-file'] = sys.argv[2]
        args['model-output'] = sys.argv[3]

        if len(sys.argv) > 4:
            args['model-input'] = sys.argv[4]

        return args
    except:
        print('Usage: text_flip_trainer.py <neural-text-file> <neural-flip-file> <model-output-file> (<model-load-file>)')

if __name__ == '__main__':
    main()
