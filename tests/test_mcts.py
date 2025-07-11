"""
This file tests key components of the MCTS and Connect Four game logic.

There are 7 tests in total:
1. test_get_valid_actions: tests that only non-full columns are returned as valid.
2. test_ucb_score_basic: tests the UCB score calculation with zero and non-zero visits.
3. test_node_expand_and_select_child: tests node expansion and child selection based on UCB.
4. test_MCTS_step_and_backprop: tests one MCTS step including value backpropagation.
5. test_MCTS_runs: tests that MCTS runs and returns a valid best action.
6. test_add_dirichlet_noise_effect: tests that adding Dirichlet noise changes child priors.
7. test_mock_model_predict: mock function to simulate model predictions for testing.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents.MCTS import (
    get_valid_actions, ucb_score, Node, MCTS_step, MCTS,
    PlayerAction, PLAYER1, PLAYER2, GameState
)

# Minimal stand-in for apply_player_action for test independence
def apply_player_action(state, action, player):
    # 1. Try to apply the player's move in the lowest available row of a column.
    # 2. If the column is full, raise ValueError.
    col = action
    for row in range(6):
        if state[row, col] == 0:
            state[row, col] = player
            return
    raise ValueError("Column full")

def mock_model_predict(state, turn):
    # 1. Simulates a model prediction returning a value and uniform-ish action probabilities.
    value = 0.5
    action_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    return value, action_probs

def test_get_valid_actions():
    # 1. Create an empty board.
    # 2. Fill column 0 completely.
    # 3. Verify column 0 is not a valid action, others are.
    state = np.zeros((6,7), dtype=int)
    state[:,0] = PLAYER1
    valid_actions = get_valid_actions(state)
    assert 0 not in valid_actions
    for col in range(1,7):
        assert col in valid_actions

def test_ucb_score_basic():
    # 1. Create dummy parent and child nodes with specified visits and values.
    # 2. Check UCB score for zero visits child equals prior score.
    # 3. Check UCB score with visits includes value term (with sign flipped).
    class DummyChild:
        def __init__(self, prior, visits, value):
            self.prior = prior
            self.visits = visits
            self.value = value
    class DummyParent:
        def __init__(self, visits):
            self.visits = visits
    parent = DummyParent(visits=10)
    child = DummyChild(prior=0.5, visits=0, value=0)
    score = ucb_score(parent, child)
    assert np.isclose(score, 0.5 * np.sqrt(10) / 1)

    child.visits = 5
    child.value = 2.0
    score = ucb_score(parent, child)
    expected_prior = 0.5 * np.sqrt(10) / 6
    expected_value = -(2.0 / 5)
    assert np.isclose(score, expected_prior + expected_value)

def test_node_expand_and_select_child():
    # 1. Initialize root node with empty board.
    # 2. Expand with two action probabilities > 0.
    # 3. Verify correct children created and child selection returns a valid child.
    state = np.zeros((6,7), dtype=int)
    root = Node(prior=1.0, turn=PLAYER1, state=state)
    action_probs = np.array([0.1, 0, 0, 0, 0, 0, 0.9])
    root.expand(action_probs)

    assert 0 in root.children
    assert 6 in root.children
    assert len(root.children) == 2

    action, child = root.select_child()
    assert action in root.children

def test_MCTS_step_and_backprop():
    # 1. Create root node.
    # 2. Run one MCTS step with a simple model predict.
    # 3. Check that root's visits and value are updated.
    state = np.zeros((6,7), dtype=int)
    root = Node(prior=1.0, turn=PLAYER1, state=state)

    def simple_model_predict(state, turn):
        val = 0.3
        probs = np.ones(7) / 7
        return val, probs

    MCTS_step(root, simple_model_predict)

    assert root.visits > 0
    assert root.value != 0

def test_MCTS_runs():
    # 1. Create root node.
    # 2. Run MCTS for 5 simulations with mock model predict.
    # 3. Assert returned best action is valid.
    state = np.zeros((6,7), dtype=int)
    root = Node(prior=1.0, turn=PLAYER1, state=state)

    best_action = MCTS(root, mock_model_predict, num_simulations=5)
    assert 0 <= best_action < 7
    assert isinstance(best_action, (int, np.integer))

def test_add_dirichlet_noise_effect():
    # 1. Create root and expand with uniform probabilities.
    # 2. Capture priors before noise.
    # 3. Add Dirichlet noise and verify priors changed.
    state = np.zeros((6,7), dtype=int)
    root = Node(prior=1.0, turn=PLAYER1, state=state)
    action_probs = np.ones(7) / 7
    root.expand(action_probs)

    priors_before = [child.prior for child in root.children.values()]
    from agents.MCTS import add_dirichlet_noise
    add_dirichlet_noise(root, alpha=0.5, epsilon=0.25)
    priors_after = [child.prior for child in root.children.values()]

    assert any(a != b for a, b in zip(priors_before, priors_after))
