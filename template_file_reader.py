import random

with open('flip_templates.txt') as flip_templates_file:
    _flip_templates = flip_templates_file.readlines()

with open('move_templates.txt') as move_templates_file:
    _move_templates = move_templates_file.readlines()

def read_flip_template(color, letter, other_color=None, other_letter=None):
    return _read_template(_flip_templates, color, letter, other_color, other_letter)

def read_move_template(color, letter, other_color=None, other_letter=None):
    return _read_template(_move_templates, color, letter, other_color, other_letter)

def _read_template(templates, color, letter, other_color, other_letter):
    rand_template = random.choice(templates)

    if ((not other_color and '%C' in rand_template)
        or (not other_letter and '%L' in rand_template)):
        return _read_template(templates, color, letter, other_color, other_letter)

    result = {}

    result['color'] = '%c' in rand_template
    result['letter'] = '%l' in rand_template
    result['text'] = rand_template.replace('%c', color)  \
                                  .replace('%l', letter) \
                                  .replace('%C', other_color) \
                                  .replace('%L', other_letter) \
                                  .replace('\n', '')
    return result
