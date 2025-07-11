"""
This file tests key utility functions used in the AlphaZero Connect Four agent.

There are 6 tests in total:
1. test_horizontal_flip_2d: tests correct flipping of a 2D board state and policy vector.
2. test_horizontal_flip_3d: tests correct flipping of a 3D state tensor and policy.
3. test_horizontal_flip_invalid_dims: tests that invalid dimensions raise ValueError.
4. test_state_to_tensor: tests tensor conversion for a given state and current player.
5. test_get_valid_actions: tests that only columns with space left are valid.
6. test_get_temperature_schedule: tests the temperature schedule for move exploration.
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from neuralnet.utils import (
    horizontal_flip,
    state_to_tensor,
    get_valid_actions,
    get_temperature
)
from game_utils import PLAYER1, PLAYER2


def test_horizontal_flip_2d():
    # 1. Create a small 2D game state and policy vector.
    # 2. Call horizontal_flip and check the flipped output matches expectations.
    state = np.array([[1, 0, 2],
                      [2, 1, 0]])
    policy = np.array([0.1, 0.7, 0.2])

    flipped_state, flipped_policy = horizontal_flip(state, policy)

    expected_state = np.array([[2, 0, 1],
                               [0, 1, 2]])
    expected_policy = np.array([0.2, 0.7, 0.1])

    np.testing.assert_array_equal(flipped_state, expected_state)
    np.testing.assert_array_equal(flipped_policy, expected_policy)

def test_horizontal_flip_3d():
    # 1. Create a 3D tensor representing channels × height × width.
    # 2. Flip and verify columns are reversed on last axis.
    state = np.arange(2 * 3 * 3).reshape((2, 3, 3))
    policy = np.array([0.3, 0.4, 0.3])

    flipped_state, flipped_policy = horizontal_flip(state, policy)

    expected_state = np.flip(state, axis=2)
    expected_policy = np.flip(policy)

    np.testing.assert_array_equal(flipped_state, expected_state)
    np.testing.assert_array_equal(flipped_policy, expected_policy)

def test_horizontal_flip_invalid_dims():
    # 1. Provide a 4D array.
    # 2. Assert ValueError is raised.
    state = np.ones((3, 3, 3, 3))
    policy = np.ones(7)
    with pytest.raises(ValueError):
        horizontal_flip(state, policy)

def test_state_to_tensor():
    # 1. Create a state with known PLAYER1 and PLAYER2 positions.
    # 2. Convert it and check each tensor channel individually.
    state = np.array([
        [0, 1, 2, 0, 0, 1, 2],
        [0, 0, 2, 0, 0, 1, 0],
        [0, 0, 0, 0, 2, 0, 0],
        [0, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0],
        [1, 2, 1, 2, 1, 2, 1]
    ])
    tensor = state_to_tensor(state, PLAYER1)

    assert tensor.shape == (3, 6, 7)
    assert np.all(tensor[0] == (state == PLAYER1).astype(np.float32))  # PLAYER1
    assert np.all(tensor[1] == (state == PLAYER2).astype(np.float32))  # PLAYER2
    assert np.all(tensor[2] == 1.0)  # Bias channel

def test_get_valid_actions():
    # 1. Create an empty board
    state = np.zeros((6, 7), dtype=int)

    # 2. Simulate partially filled columns, and make column 3 full
    state[:, 3] = [1, 2, 1, 2, 1, 2]  # Fully filled column 3

    # 3. Run get_valid_actions and check results
    valid = get_valid_actions(state)
    assert 3 not in valid
    for col in [0, 1, 2, 4, 5, 6]:
        assert col in valid

def test_get_temperature_schedule():
    # 1. Check exploration temperature at different move stages.
    assert get_temperature(0) == 1.2      # Start
    assert get_temperature(20) == 1.2     # Early game
    assert get_temperature(25) == 0.6     # Mid game
    assert get_temperature(35) == 0.1     # Late game
    assert get_temperature(41) == 0.0     # End game
