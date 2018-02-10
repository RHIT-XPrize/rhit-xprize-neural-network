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

    (text_list, soln_list) = build_list(args['num-to-generate'], colors, letters)

    soln_list = expand_soln_list(soln_list, len(colors), len(letters))

    write_csv(args['neural-infile'], text_list)
    write_csv(args['neural-outfile'], soln_list)

def get_args():
    try:
        assert len(sys.argv) == 4

        args = {}

        args['num-to-generate'] = int(sys.argv[1])
        args['neural-infile'] = sys.argv[2]
        args['neural-outfile'] = sys.argv[3]

        return args
    except:
        print('Call as "python generate_text_instructions.py <num-to-generate> <neural-infile> <neural-outfile>"')

def load_colors():
    with open('colors.txt') as colors_file:
        return list(map(lambda color: color.replace('\n', ''), colors_file.readlines()))

def load_letters():
    letter_nums = list(range(ord('A'), ord('H') + 1))
    return list(map(lambda i: chr(i), letter_nums))

def build_list(num_to_generate, colors, letters):
    text_list = []
    soln_list = []

    for _ in range(num_to_generate):
        soln = {}
        if should_flip():
            template_func = reader.read_flip_template
            soln['flipped'] = True
        else:
            template_func = reader.read_move_template
            soln['flipped'] = False

        color_ind = random.randint(0, len(colors) - 1)
        rand_color = colors[color_ind]

        letter_ind = random.randint(0, len(letters) - 1)
        rand_letter = letters[letter_ind]

        text_list += [[template_func(rand_color, rand_letter).upper()]]

        soln['color-ind'] = color_ind
        soln['letter-ind'] = letter_ind

        soln_list += [soln]

    return (text_list, soln_list)

def should_flip():
    return random.random() < FLIP_PERCENTAGE

def write_csv(filename, data_list):
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)

        for data in data_list:
            csv_writer.writerow(data)

def expand_soln_list(soln_list, num_colors, num_letters):
    def expand_soln(soln):
        colors_out = [0] * num_colors
        letters_out = [0] * num_letters

        colors_out[soln['color-ind']] = 1
        letters_out[soln['letter-ind']] = 1
        flipped = [1 if soln['flipped'] else 0]

        return flipped + colors_out + letters_out

    return list(map(expand_soln, soln_list))

if __name__ == '__main__':
    main()
