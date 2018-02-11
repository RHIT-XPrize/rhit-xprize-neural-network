import csv
import random
import sys

import template_file_reader as reader

FLIP_PERCENTAGE = 0.3

def main():
    args = get_args()

    if not args:
        return

    colors = load_colors()
    letters = load_letters()

    data = build_data(args['num-to-generate'], colors, letters)

    data = expand_data(data, len(colors), len(letters))

    write_csv(args['neural-infile'], data['text'])
    write_csv(args['neural-flips-outfile'], data['flipped'])
    write_csv(args['neural-color-outfile'], data['colors'])
    write_csv(args['neural-letter-outfile'], data['letters'])

def get_args():
    try:
        assert len(sys.argv) == 6

        args = {}

        args['num-to-generate'] = int(sys.argv[1])
        args['neural-infile'] = sys.argv[2]
        args['neural-flips-outfile'] = sys.argv[3]
        args['neural-color-outfile'] = sys.argv[4]
        args['neural-letter-outfile'] = sys.argv[5]

        return args
    except:
        print('Call as "python generate_text_instructions.py <num-to-generate> <neural-infile> <neural-flips-outfile> <neural-color-outfile> <neural-letter-outfile>"')

def load_colors():
    with open('colors.txt') as colors_file:
        return list(map(lambda color: color.replace('\n', ''), colors_file.readlines()))

def load_letters():
    letter_nums = list(range(ord('A'), ord('H') + 1))
    return list(map(lambda i: chr(i), letter_nums))

def build_data(num_to_generate, colors, letters):
    data = {
        'text': [],
        'flipped': [],
        'colors': [],
        'letters': []
    }

    for _ in range(num_to_generate):
        if should_flip():
            template_func = reader.read_flip_template
            data['flipped'] += [True]
        else:
            template_func = reader.read_move_template
            data['flipped'] += [False]

        (color_ind, rand_color, other_color) = random_elements(colors)

        (letter_ind, rand_letter, other_letter) = random_elements(letters)

        template_result = template_func(rand_color, rand_letter,
                                        other_color, other_letter)
        data['text'] += [[template_result['text'].upper()]]

        if template_result['color']:
            data['colors'] += [color_ind]
        else:
            data['colors'] += [0]

        if template_result['letter']:
            data['letters'] += [letter_ind]
        else:
            data['letters'] += [0]

    return data

def random_elements(elements):
    rand_index = random.randint(0, len(elements) - 1)
    rand_element = elements[rand_index]

    # Pick a different one, but don't get stuck
    for _ in range(10):
        other_element = random.choice(elements)
        if other_element != rand_element:
            break
        else:
            other_element = None

    return (rand_index, rand_element, other_element)

def should_flip():
    return random.random() < FLIP_PERCENTAGE

def write_csv(filename, data_list):
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)

        for data in data_list:
            csv_writer.writerow(data)

def expand_data(data, num_colors, num_letters):
    def expand_flipped(flipped):
        for row_index, flipped_entry in enumerate(flipped):
            flipped[row_index] = [1 if flipped_entry else 0]

        return flipped

    def expand_colors(colors):
        for row_index, color_index in enumerate(colors):
            colors_out = [0] * num_colors
            colors_out[color_index] = 1
            colors[row_index] = colors_out

        return colors

    def expand_letters(letters):
        for row_index, letter_index in enumerate(letters):
            letters_out = [0] * num_letters

            letters_out[letter_index] = 1
            letters[row_index] = letters_out

        return letters

    return {
        'text': data['text'],
        'flipped': expand_flipped(data['flipped']),
        'colors': expand_colors(data['colors']),
        'letters': expand_letters(data['letters'])
    }

if __name__ == '__main__':
    main()
