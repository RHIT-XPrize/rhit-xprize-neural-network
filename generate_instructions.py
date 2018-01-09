from random import randint, random


"""
Classes
"""


class NoActionException(Exception):
    pass


class Block:
    last_id = 0

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
        return self.block_id == other.block_id

    def __ne__(self, other):
        return not self == other

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

    def __str__(self):
        return '{side1: (%s, %s), side2: (%s, %s), pos: %s, id: %s}' % \
            (self.side1_letter, self.side1_color,
             self.side2_letter, self.side2_color,
             self.position, self.block_id)


class Instruction:
    def __init__(self, phrase, point):
        self.phrase = phrase
        self.point = point

    def __str__(self):
        return self.phrase


class Configuration:
    def __init__(self, blocks, final_blocks=[]):
        self.current_blocks = blocks
        self.final_blocks = final_blocks

    def is_complete(self):
        return self.current_blocks == []

    @staticmethod
    def get_instruction(moved_block):
        point = moved_block.position
        phrase = 'move the %s %s block here' \
                 % (moved_block.side1_color,
                    moved_block.side1_letter)
        return Instruction(phrase, point)

    def generate_action(self, goal_config):
        if self.is_complete():
            raise NoActionException('Board is already complete')

        block_to_move = rand_element(self.current_blocks)
        moved_block_index = goal_config.final_blocks.index(block_to_move)
        moved_block = goal_config.final_blocks[moved_block_index]
        new_current_blocks = self.current_blocks[:]
        new_current_blocks.remove(moved_block)
        new_configuration = Configuration(
            new_current_blocks,
            self.final_blocks + [moved_block]
        )
        instruction = Configuration.get_instruction(moved_block)
        return Action(self, new_configuration, instruction)

    def scatter(self):
        return Configuration(list(map(randomize_block,
                                      self.current_blocks + self.final_blocks)))

    def __str__(self):
        # :(
        return str(list(map(str, self.current_blocks + self.final_blocks)))

    def mark_complete(self):
        return Configuration([], self.current_blocks + self.final_blocks)


class Action:
    def __init__(self, start_conf, end_conf, phrase):
        self.start_conf = start_conf
        self.end_conf = end_conf
        self.phrase = phrase

    def __str__(self):
        return str(self.phrase)


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
        # if random() < 0.0:
        #     return b.flip()
        return b
    def maybe_shift(b):
        if random() < 0.95:
            return b.shift_to(random_position())
        return b
    return maybe_flip(maybe_shift(block))


def random_configuration(num_blocks, letters, colors):
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

def main():
    colors = ['RED', 'GREEN', 'BLUE']
    letters = ['A', 'B', 'C']
    start = random_configuration(2, letters, colors)
    end = start.scatter().mark_complete()

    b = random_block(colors, letters)
    b_prime = b.shift_to(random_position())

    print('start: %s' % start)
    print('end: %s' % end)
    print()
    print('actions: %s' % str(list(map(str, solve_board(start, end)))))


if __name__ == '__main__':
    main()
