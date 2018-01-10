import generate_instructions as gi

"""
Block tests
"""

def test_block_inequality_with_generated_ids():
    assert gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0)) \
        != gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))

def test_block_equality_with_given_ids():
    assert gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0), 30) \
        == gi.Block('D', 'YELLOW', 'C', 'ORANGE', (-1, 2), 30)

def test_block_shift_to_keeps_id():
    base_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))
    shifted_block = base_block.shift_to((2, 3))

    assert base_block.block_id == shifted_block.block_id

def test_block_shift_to_keeps_visuals():
    base_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))
    shifted_block = base_block.shift_to((2, 3))

    assert base_block.side1_letter == shifted_block.side1_letter
    assert base_block.side1_color == shifted_block.side1_color
    assert base_block.side2_letter == shifted_block.side2_letter
    assert base_block.side2_color == shifted_block.side2_color

def test_block_shift_to_updates_pos():
    new_pos = (2, 3)
    base_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))
    shifted_block = base_block.shift_to(new_pos)

    assert shifted_block.position == new_pos

def test_block_flip_keeps_block_id():
    base_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))
    flipped_block = base_block.flip()

    assert base_block.block_id == flipped_block.block_id


def test_block_flip_keeps_pos():
    base_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))
    flipped_block = base_block.flip()

    assert base_block.position == flipped_block.position

def test_block_flip_swaps_visuals():
    base_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))
    flipped_block = base_block.flip()

    assert base_block.side1_letter == flipped_block.side2_letter
    assert base_block.side1_color == flipped_block.side2_color
    assert base_block.side2_letter == flipped_block.side1_letter
    assert base_block.side2_color == flipped_block.side1_color


"""
Configuration
"""

def test_is_complete_empty_current_empty_final():
    assert gi.Configuration([], []).is_complete()

def test_is_complete_nonempty_current_empty_final():
    assert not gi.Configuration([1], []).is_complete()

def test_is_complete_nonempty_current_nonempty_final():
    assert not gi.Configuration([1], [2]).is_complete()

def test_get_instruction_phrase():
    moved_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 2))
    instruction = gi.Configuration.get_instruction(moved_block)

    assert 'A' in instruction.phrase
    assert 'BLUE' in instruction.phrase

def test_get_instruction_point():
    moved_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 2))
    instruction = gi.Configuration.get_instruction(moved_block)

    assert instruction.point == (1, 2)

def test_get_moved_block():
    start_conf = gi.random_configuration(1, ['A'], ['red'])
    end_conf = start_conf.scatter().mark_complete()
    [action] = gi.solve_board(start_conf, end_conf)

    moved_block = action.get_moved_block()
    assert moved_block.side1_letter == 'A'
    assert moved_block.side1_color == 'red'

# it is criminal that this is not built in
def list_contains(longer, shorter):
    if shorter == []:
        return True
    if longer == []:
        return False
    if shorter[0] == longer[0]:
        return list_contains(longer[1:], shorter[1:]) or \
            list_contains(longer[1:], shorter)
    return list_contains(longer[1:], shorter)

def test_list_contains():
    assert list_contains([], [])
    assert list_contains([1], [])
    assert list_contains([1], [1])
    assert list_contains([1, 2, 3, 4], [2, 3])
    assert not list_contains([], [2, 3])
    assert not list_contains([1, 2], [2, 3])
    assert not list_contains([1, 3, 2], [2, 3])

class TestListRepresentations():
    def setup_method(self):
        self.start_conf = gi.random_configuration(2, ['A'], ['red'])
        self.end_conf = self.start_conf.scatter().mark_complete()
        self.block = self.start_conf.current_blocks[0]

    def test_block_list_representation(self):
        block = gi.random_block(['A'], ['red'])
        [_, letter1, color1, letter2, color2, x, y] = block.list_representation()

        assert block.side1_letter == letter1
        assert block.side1_color == color1
        assert block.side2_letter == letter2
        assert block.side2_color == color2
        assert block.position[0] == x
        assert block.position[1] == y

    def test_instruction_list_representation(self):
        inst = gi.Instruction('foobar', (1, 2))
        [phrase, x, y] = inst.list_representation()

        assert 'foobar' == phrase
        assert 1 == x
        assert 2 == y

    def test_configuration_list_representation(self):
        start_conf = gi.random_configuration(2, ['A', 'B'], ['red', 'green'])
        list_repr = start_conf.list_representation()

        block1 = start_conf.get_all_blocks()[0]
        block2 = start_conf.get_all_blocks()[1]

        b1_list_repr = block1.list_representation()
        b2_list_repr = block2.list_representation()

        assert list_repr == b1_list_repr + b2_list_repr

    def test_action_list_representation(self):
        start_conf = gi.random_configuration(2, ['A', 'B'], ['red', 'green'])
        end_conf = start_conf.scatter().mark_complete()
        actions = gi.solve_board(start_conf, end_conf)

        action = actions[0]
        moved_block = action.get_moved_block()
        inst = action.instruction
        list_repr = action.list_representation()

        assert list_repr == start_conf.list_representation() + \
            [moved_block.block_id,
             moved_block.position[0],
             moved_block.position[1]] + \
             inst.list_representation()


class TestGenerateAction:
    def setup_method(self):
        self.moved_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 2), 1)
        self.destination_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (2, 2), 1)

        self.goal_block = gi.Block('B', 'YELLOW', 'G', 'RED', (3, 3), 2)
        self.goal_config = gi.Configuration([], [self.destination_block, self.goal_block])

        self.instruction = gi.Instruction('Phrase', (2, 2))

        def mock_rand_element(blocks):
            return self.moved_block

        self.orig_rand_element = gi.rand_element
        gi.rand_element = mock_rand_element

        def mock_get_instruction(block):
            return self.instruction

        self.orig_get_instruction = gi.Configuration.get_instruction
        gi.Configuration.get_instruction = mock_get_instruction

    def teardown_method(self):
        gi.rand_element = self.orig_rand_element
        gi.Configuration.get_instruction = self.orig_get_instruction

    def test_generate_action_complete_board(self):
        finished_configuration = gi.Configuration([])
        try:
            finished_configuration.generate_action(0)
            assert False
        except gi.NoActionException:
            assert True

    def test_generate_action_completing_board(self):
        base_config = gi.Configuration([self.moved_block], [self.goal_block])
        action = base_config.generate_action(self.goal_config)

        assert action.start_conf == base_config
        assert action.instruction == self.instruction

        self.assert_action_current_blocks(action)
        self.assert_action_final_blocks(action)

    def test_generate_action_not_completing_board(self):
        unmoved_block = gi.Block('B', 'YELLOW', 'G', 'RED', (3, 3), 2)
        base_config = gi.Configuration([self.moved_block, unmoved_block], [])

        action = base_config.generate_action(self.goal_config)

        assert action.start_conf == base_config
        assert action.instruction == self.instruction

        self.assert_action_current_blocks(action)
        self.assert_action_final_blocks(action)

    def assert_action_current_blocks(self, action):
        assert len(action.end_conf.current_blocks) \
            == len(action.start_conf.current_blocks) - 1

        for end_current_block in action.end_conf.current_blocks:
            assert end_current_block in action.start_conf.current_blocks

        assert self.moved_block not in action.end_conf.current_blocks

    def assert_action_final_blocks(self, action):
        assert len(action.end_conf.final_blocks) \
            == len(action.start_conf.final_blocks) + 1

        for start_goal_block in action.start_conf.final_blocks:
            assert start_goal_block in action.end_conf.final_blocks

        assert self.destination_block in action.end_conf.final_blocks

class TestScatter:
    def setup_method(self):
        self.unscattered_blocks = [gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 1), 1),
                                   gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 1), 2),
                                   gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 1), 3),
                                   gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 1), 4)]

        self.scattered_blocks = [gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0), 1),
                                 gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0), 2),
                                 gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0), 3),
                                 gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0), 4)]
        self.randomizer_index = 0

        def mock_randomize_block(block):
            to_return = self.scattered_blocks[self.randomizer_index]
            self.randomizer_index += 1
            return to_return

        self.orig_randomize_block = gi.randomize_block
        gi.randomize_block = mock_randomize_block

    def teardown_method(self):
        gi.randomize_block = self.orig_randomize_block

    def test_scatter_empty(self):
        config = gi.Configuration([])
        scattered_config = config.scatter()

        assert scattered_config.current_blocks == []
        assert scattered_config.final_blocks == []

    def test_scatter_only_current(self):
        base_config = gi.Configuration(self.unscattered_blocks[:])
        scattered_config = base_config.scatter()

        assert scattered_config.final_blocks == []
        assert scattered_config.current_blocks == self.scattered_blocks

    def test_scatter_current_and_final(self):
        base_config = gi.Configuration(self.unscattered_blocks[0:2],
                                       self.unscattered_blocks[2:4])
        scattered_config = base_config.scatter()

        assert scattered_config.final_blocks == []
        assert scattered_config.current_blocks == self.scattered_blocks

def test_random_block():
    indexer = {}
    indexer['rand_index'] = 0

    random_position = (23, 54)

    letters = ['A', 'B', 'C', 'D']
    colors = ['BLUE', 'GREEN', 'RED', 'YELLOW']
    block = gi.random_block(letters, colors)

    assert block.side1_letter in letters
    assert block.side2_letter in letters
    assert block.side1_color in colors
    assert block.side2_color in colors
    assert block.position[0] > 0
    assert block.position[1] > 0

class TestSolveBoard:
    # TODO: Add test for flips once implemented
    def setup_method(self):
        initial_config = gi.Configuration([
            gi.Block('A', 'BLUE', 'B', 'YELLOW', (0, 1), 1),
            gi.Block('C', 'GREEN', 'D', 'RED', (1, 2), 2),
            gi.Block('E', 'ORANGE', 'F', 'PURPLE', (2, 3), 3)])
        self.goal_config = gi.Configuration([], [
            gi.Block('A', 'BLUE', 'B', 'YELLOW', (0, 2), 1),
            gi.Block('C', 'GREEN', 'D', 'RED', (1, 3), 2),
            gi.Block('E', 'ORANGE', 'F', 'PURPLE', (2, 4), 3)])
        self.actions = gi.solve_board(initial_config, self.goal_config)

    def test_solve_board_length(self):
        assert len(self.actions) == 3

    def test_solve_board_consecutive_configurations(self):
        for index, action in enumerate(self.actions[0:-2]):
            next_action = self.actions[index + 1]

            assert action.end_conf == next_action.start_conf

    def test_solve_board_single_changes(self):
        for action in self.actions:
            assert len(action.end_conf.final_blocks) \
                == len(action.start_conf.final_blocks) + 1
            assert len(action.start_conf.current_blocks) \
                == len(action.end_conf.current_blocks) + 1

    def test_solve_board_final_action_yields_goal(self):
        final_config = self.actions[-1].end_conf

        assert final_config.current_blocks == []

        for block in final_config.final_blocks:
            assert block in self.goal_config.final_blocks
        for block in self.goal_config.final_blocks:
            assert block in final_config.final_blocks
