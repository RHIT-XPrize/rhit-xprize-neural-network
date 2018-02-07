from random import randint, random
import sys
import csv

import template_file_reader as tr

"""
Classes
"""


class NoActionException(Exception):
    pass

class InvalidActionException(Exception):
    pass


class Block:
    last_id = 0

    @staticmethod
    def reset_block_id():
        Block.last_id = 0

    @staticmethod
    def get_next_block_id():
        Block.last_id += 1
        return Block.last_id

    def __init__(self, side1_letter,
                 side1_color,
                 side2_letter,
                 side2_color,
                 pos,
                 block_id=-1):
        self.side1_letter = side1_letter
        self.side1_color = side1_color
        self.side2_letter = side2_letter
        self.side2_color = side2_color
        self.position = pos
        self.block_id = block_id if block_id > 0 else Block.get_next_block_id()

    def __eq__(self, other):
        return type(other) == type(self) and self.block_id == other.block_id

    def __lt__(self, other):
        return self.block_id < other.block_id

    def looks_the_same(self, other):
        return self.side1_letter == other.side1_letter and \
            self.side1_color == other.side1_color and \
            self.side2_letter == other.side2_letter and \
            self.side2_color == other.side2_color

    def shift_to(self, position):
        return Block(
            self.side1_letter,
            self.side1_color,
            self.side2_letter,
            self.side2_color,
            position,
            self.block_id
        )

    def flip(self):
        return Block(
            self.side2_letter,
            self.side2_color,
            self.side1_letter,
            self.side1_color,
            self.position,
            self.block_id
        )

    def list_representation(self):
        return [
            self.block_id,
            self.side1_letter,
            self.side1_color,
            self.side2_letter,
            self.side2_color,
            self.position[0],
            self.position[1],
        ]

    def __str__(self):
        return '{side1: (%s, %s), side2: (%s, %s), pos: %s, id: %s}' % \
            (self.side1_letter, self.side1_color,
             self.side2_letter, self.side2_color,
             self.position, self.block_id)

def create_move_instruction(block):
    point = block.position
    phrase = tr.read_move_template(block.side1_color, block.side1_letter)
    return Instruction(phrase, point)

def create_flip_instruction(block):
    phrase = tr.read_flip_template(block.side1_color, block.side1_letter)
    return Instruction(phrase, block.position)

class Instruction:
    def __init__(self, phrase, point):
        self.phrase = phrase
        self.point = point

    def list_representation(self):
        return [self.phrase, self.point[0], self.point[1]]

    def __str__(self):
        return self.phrase


class Configuration:
    def __init__(self, blocks, final_blocks=[]):
        self.current_blocks = blocks
        self.final_blocks = final_blocks

    def is_complete(self):
        return self.current_blocks == []

    def _get_next_config_and_block(self, goal_block, should_flip_block, block_to_move):
        new_current_blocks = self.current_blocks[:]
        new_final_blocks = self.final_blocks
        if should_flip_block:
            moved_block = block_to_move.flip()
            new_current_blocks.remove(block_to_move)
            new_current_blocks.append(moved_block)
        else:
            moved_block = goal_block
            new_current_blocks.remove(block_to_move)
            new_final_blocks = new_final_blocks + [moved_block]

        return moved_block, Configuration(
            new_current_blocks,
            new_final_blocks
        )

    def generate_action(self, goal_config):
        if self.is_complete():
            raise NoActionException('Board is already complete')

        block_to_move = rand_element(self.current_blocks)

        goal_block_index = goal_config.final_blocks.index(block_to_move)
        goal_block = goal_config.final_blocks[goal_block_index]

        should_flip_block = not block_to_move.looks_the_same(goal_block)
        (moved_block, new_configuration) = self._get_next_config_and_block(
            goal_block,
            should_flip_block,
            block_to_move
        )

        if should_flip_block:
            instruction = create_flip_instruction(moved_block)
        else:
            instruction = create_move_instruction(moved_block)

        return Action(self, new_configuration, instruction)

    def scatter(self):
        return Configuration(list(map(randomize_block,
                                      self.get_all_blocks())))

    def list_representation(self):
        all_blocks = self.get_all_blocks()
        return [x for b in all_blocks for x in b.list_representation()]

    def __str__(self):
        return str(list(map(str, self.get_all_blocks())))

    def mark_complete(self):
        return Configuration([], self.get_all_blocks())

    def get_all_blocks(self):
        return self.current_blocks + self.final_blocks

    def __eq__(self, other):
        eq_helper = lambda ls1, ls2: all([x == y and x.looks_the_same(y) \
                                     for (x, y) in zip(sorted(ls1), sorted(ls2))])

        return eq_helper(self.current_blocks, other.current_blocks) and \
            eq_helper(self.final_blocks, other.final_blocks)


class Action:
    def __init__(self, start_conf, end_conf, instruction):
        self.start_conf = start_conf
        self.end_conf = end_conf
        self.instruction = instruction

    def _get_translated_block(self):
        for block in self.end_conf.final_blocks:
            if not block in self.start_conf.final_blocks:
                return block

    def _get_flipped_block(self):
        for block in self.end_conf.current_blocks:
            for other_block in self.start_conf.current_blocks:
                if (other_block == block and
                    other_block.flip().looks_the_same(block)):
                    return block

    def get_moved_block(self):
        moved_block = self._get_translated_block() or self._get_flipped_block()
        if moved_block:
            return moved_block
        raise InvalidActionException('Action has no moved block')

    def list_representation(self):
        start_list_repr = self.start_conf.list_representation()
        moved_block = self.get_moved_block()
        return start_list_repr + [
            moved_block.block_id,
            moved_block.position[0],
            moved_block.position[1],
        ] + self.instruction.list_representation()

    def __str__(self):
        return str(self.instruction)


"""
Randomness
"""

def rand_element(ls):
    return ls[rand_index(ls)]


def rand_index(ls):
    return randint(0, len(ls) - 1)


def random_position():
    return (randint(0, 1000), randint(0, 1000))


def random_block(letters, colors):
    side1_letter = rand_element(letters)
    side2_letter = rand_element(letters)
    side1_color  = rand_element(colors)
    side2_color  = rand_element(colors)
    position = random_position()
    return Block(
        side1_letter,
        side1_color,
        side2_letter,
        side2_color,
        position
    )

def randomize_block(block):
    def maybe_flip(b):
        if random() < 0.5:
            return b.flip()
        return b
    def maybe_shift(b):
        if random() < 0.95:
            return b.shift_to(random_position())
        return b
    return maybe_flip(maybe_shift(block))


def random_configuration(num_blocks, letters, colors):
    Block.reset_block_id()
    blocks = [random_block(letters, colors) for i in range(num_blocks)]
    return Configuration(blocks)


"""
Entry Points
"""

def solve_board(current_config, goal_config):
    try:
        action = current_config.generate_action(goal_config)
        rest = solve_board(action.end_conf, goal_config)
        return [action] + rest
    except NoActionException as e:
        return []

def create_header_list(num_blocks):
    header_ls = []
    for i in range(num_blocks):
        header_ls.append('block{0}_id'.format(i))
        header_ls.append('block{0}_side1_letter'.format(i))
        header_ls.append('block{0}_side1_color'.format(i))
        header_ls.append('block{0}_side2_letter'.format(i))
        header_ls.append('block{0}_side2_color'.format(i))
        header_ls.append('block{0}_x_pos'.format(i))
        header_ls.append('block{0}_y_pos'.format(i))
    header_ls.append('moved_block_id')
    header_ls.append('moved_block_x_pos')
    header_ls.append('moved_block_y_pos')
    header_ls.append('phrase')
    header_ls.append('point_x')
    header_ls.append('point_y')
    return header_ls

def main():
    colors = ['RED', 'GREEN', 'BLUE']
    letters = ['A', 'B', 'C']

    if len(sys.argv) != 4:
        print('Wrong number of arguments.')
        print('Call as "python generate_instructions.py <num-to-generate> <num-blocks> <outfile>"')
        return

    try:
        num_to_generate = int(sys.argv[1])
    except:
        print('Number of games must be an integer.')
        return

    try:
        num_blocks = int(sys.argv[2])
    except:
        print('Number of blocks must be an integer')

    try:
        f = open(sys.argv[3], 'w')
        csv_writer = csv.writer(f)
    except:
        print('Given path is not valid or could not be opened')
        return

    csv_writer.writerow(create_header_list(num_blocks))
    for _ in range(num_to_generate):
        start = random_configuration(num_blocks, letters, colors)
        end = start.scatter().mark_complete()
        moves = solve_board(start, end)
        for line_repr in map(lambda x: x.list_representation(), moves):
            csv_writer.writerow(line_repr)

    f.close()


if __name__ == '__main__':
    main()
