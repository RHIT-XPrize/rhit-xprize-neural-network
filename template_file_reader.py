import random

with open('flip_templates.txt') as flip_templates_file:
    _flip_templates = flip_templates_file.readlines()

with open('move_templates.txt') as move_templates_file:
    _move_templates = move_templates_file.readlines()

def read_flip_template(color, letter):
    return _read_template(_flip_templates, color, letter)

def read_move_template(color, letter):
    return _read_template(_move_templates, color, letter)

def _read_template(templates, color, letter):
    rand_template = random.choice(templates)

    result = {}

    result['color'] = '%c' in rand_template
    result['letter'] = '%l' in rand_template
    result['text'] = rand_template.replace('%c', color)  \
                                  .replace('%l', letter) \
                                  .replace('\n', '')
    return result
