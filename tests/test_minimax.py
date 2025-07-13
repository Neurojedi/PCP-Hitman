"""
This file tests key components of the Minimax-based Connect Four agent.

There are 7 tests in total:
1. test_evaluate_board_piece_diff: tests piece difference heuristic evaluation.
2. test_evaluate_board_center_weight: tests center column weight heuristic evaluation.
3. test_evaluate_board_window_count_win: tests window count heuristic detecting a winning state.
4. test_evaluate_board_window_count_three_in_row: tests window count heuristic detecting three in a row.
5. test_minimax_basic: tests basic minimax score and column selection with mocked game logic.
6. test_generate_move_minimax_returns_valid_move: tests that generate_move_minimax returns a valid column under normal conditions.
7. test_generate_move_minimax_fallback: tests that generate_move_minimax falls back to a random valid column if minimax detects a terminal state.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from agents.minimax import evaluate_board, minimax, generate_move_minimax
from game_utils import PLAYER1, PLAYER2, NO_PLAYER, PlayerAction, GameState

# Helper: create an empty board
def empty_board():
    return np.full((6, 7), NO_PLAYER, dtype=np.int8)

def test_evaluate_board_piece_diff():
    board = empty_board()
    board[5, 3] = PLAYER1
    board[4, 3] = PLAYER2
    score = evaluate_board(board, PLAYER1, heuristic="piece_diff")
    assert score == 0  # 1 piece each => 1 - 1 == 0

def test_evaluate_board_center_weight():
    board = empty_board()
    center_col = 3
    board[5, center_col] = PLAYER1
    board[4, center_col] = PLAYER1
    board[3, center_col] = PLAYER2
    score = evaluate_board(board, PLAYER1, heuristic="center_weight")
    # PLAYER1: 2 pieces * 2 = 4; PLAYER2: 1 piece * 2 = 2; total = 4 - 2 = 2
    assert score == 2

def test_evaluate_board_window_count_win():
    board = empty_board()
    # PLAYER1 wins horizontally
    board[5, 0:4] = PLAYER1
    score = evaluate_board(board, PLAYER1, heuristic="window_count")
    assert score >= 1e6

def test_evaluate_board_window_count_three_in_row():
    board = empty_board()
    # Three in a row for PLAYER1 horizontally
    board[5, 0:3] = PLAYER1
    score = evaluate_board(board, PLAYER1, heuristic="window_count")
    assert score >= 5.0

# Mock helpers for minimax tests
def mock_check_end_state(board, player):
    # No one has won, no draw for this test
    return GameState.NOT_OVER

def mock_apply_player_action(board, action, player):
    # Put player's piece at the bottom-most empty slot in column
    col = action.column
    for r in reversed(range(board.shape[0])):
        if board[r, col] == NO_PLAYER:
            board[r, col] = player
            break

def test_minimax_basic(monkeypatch):
    # Patch external functions
    monkeypatch.setattr("game_utils.check_end_state", mock_check_end_state)
    monkeypatch.setattr("game_utils.apply_player_action", mock_apply_player_action)

    board = empty_board()
    score, col = minimax(board, PLAYER1, depth=1, alpha=float("-inf"), beta=float("inf"), maximizing=True, heuristic="piece_diff")
    assert isinstance(score, float)
    assert col in range(board.shape[1])

def test_generate_move_minimax_returns_valid_move(monkeypatch):
    monkeypatch.setattr("game_utils.check_end_state", mock_check_end_state)
    monkeypatch.setattr("game_utils.apply_player_action", mock_apply_player_action)

    board = empty_board()
    action, saved = generate_move_minimax(board, PLAYER1, saved_state=None, depth=2, heuristic="piece_diff")
    assert isinstance(action, PlayerAction)
    assert 0 <= int(action) < board.shape[1]

def test_generate_move_minimax_fallback(monkeypatch):
    # Patch check_end_state to always return a terminal win to force minimax to return None
    def end_state_win(board, player):
        return GameState.IS_WIN

    monkeypatch.setattr("game_utils.check_end_state", end_state_win)
    monkeypatch.setattr("game_utils.apply_player_action", mock_apply_player_action)

    board = empty_board()
    action, saved = generate_move_minimax(board, PLAYER1, saved_state=None, depth=2, heuristic="piece_diff")
    # Should pick a valid column anyway (random fallback)
    assert isinstance(action, PlayerAction)
    assert 0 <= int(action) < board.shape[1]
