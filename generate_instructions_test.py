import generate_instructions as gi
import string

"""
Block tests
"""

def test_block_inequality_with_generated_ids():
    assert gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0)) \
        != gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0))

def test_block_equality_with_given_ids():
    assert gi.Block('A', 'BLUE', 'B', 'GREEN', (0, 0), 30) \
        == gi.Block('D', 'YELLOW', 'C', 'ORANGE', (-1, 2), 30)

def test_block_looks_the_same():
    block = gi.Block('A', 'Blue', 'B', 'Red', (0, 0))

    assert block.looks_the_same(block)
    assert block.looks_the_same(gi.Block('A', 'Blue', 'B', 'Red', (0, 0)))
    assert block.looks_the_same(gi.Block('A', 'Blue', 'B', 'Red', (99, 99)))
    assert not block.looks_the_same(gi.Block('A', 'Blue', 'B', 'Green', (0, 0)))

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

def test_random_block():
    letters = ['A', 'B', 'C', 'D']
    colors = ['BLUE', 'GREEN', 'RED', 'YELLOW']
    block = gi.random_block(letters, colors)

    assert block.side1_letter in letters
    assert block.side2_letter in letters
    assert block.side1_color in colors
    assert block.side2_color in colors
    assert block.position[0] > 0
    assert block.position[1] > 0


"""
Configuration
"""

def test_is_complete_empty_current_empty_final():
    assert gi.Configuration([], []).is_complete()

def test_is_complete_nonempty_current_empty_final():
    assert not gi.Configuration([1], []).is_complete()

def test_is_complete_nonempty_current_nonempty_final():
    assert not gi.Configuration([1], [2]).is_complete()

def test_get_move_instruction_phrase():
    moved_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 2))
    instruction = gi.create_move_instruction(moved_block)

    assert 'A' in instruction.phrase
    assert 'BLUE' in instruction.phrase

def test_get_instruction_point():
    moved_block = gi.Block('A', 'BLUE', 'B', 'GREEN', (1, 2))
    instruction = gi.create_move_instruction(moved_block)

    assert instruction.point == (1, 2)

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

        self.orig_get_instruction = gi.create_move_instruction
        gi.create_move_instruction = mock_get_instruction

    def teardown_method(self):
        gi.rand_element = self.orig_rand_element
        gi.create_move_instruction = self.orig_get_instruction

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
        assert action.phrase == self.instruction

        self.assert_action_current_blocks(action)
        self.assert_action_final_blocks(action)

    def test_generate_action_not_completing_board(self):
        unmoved_block = gi.Block('B', 'YELLOW', 'G', 'RED', (3, 3), 2)
        base_config = gi.Configuration([self.moved_block, unmoved_block], [])

        action = base_config.generate_action(self.goal_config)

        assert action.start_conf == base_config
        assert action.phrase == self.instruction

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

class TestSolveBoard:
    def setup_method(self):
        initial_config = gi.Configuration([
            gi.Block('A', 'BLUE', 'B', 'YELLOW', (0, 1), 1),
            gi.Block('C', 'GREEN', 'D', 'RED', (1, 2), 2),
            gi.Block('E', 'ORANGE', 'F', 'PURPLE', (2, 3), 3)])
        self.goal_config = gi.Configuration([], [
            gi.Block('A', 'BLUE', 'B', 'YELLOW', (0, 2), 1),
            gi.Block('D', 'RED', 'C', 'GREEN', (1, 3), 2),
            gi.Block('E', 'ORANGE', 'F', 'PURPLE', (2, 4), 3)])
        self.actions = gi.solve_board(initial_config, self.goal_config)

    def test_solve_board_length(self):
        assert len(self.actions) == 4

    def test_solve_board_consecutive_configurations(self):
        for index, action in enumerate(self.actions[0:-2]):
            next_action = self.actions[index + 1]

            assert action.end_conf == next_action.start_conf

    def test_solve_board_final_action_yields_goal(self):
        final_config = self.actions[-1].end_conf

        assert final_config.current_blocks == []

        for block in final_config.final_blocks:
            assert block in self.goal_config.final_blocks
        for block in self.goal_config.final_blocks:
            assert block in final_config.final_blocks

    def test_solve_board_stress(self):
        num_blocks = 500
        letters = string.ascii_uppercase
        colors = ['Red', 'Green', 'Blue', 'Gold', 'Brown', 'Purple', 'Yellow', 'Pink']

        start_config = gi.random_configuration(num_blocks, letters, colors)
        final_config = start_config.scatter().mark_complete()

        actions = gi.solve_board(start_config, final_config)
        solved_config = actions[len(actions) - 1].end_conf

        assert final_config == solved_config
