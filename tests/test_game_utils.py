import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from game_utils import (
    initialize_game_state, pretty_print_board, string_to_board,
    apply_player_action, check_move_status, check_end_state,
    connected_four, GameState, MoveStatus,
    NO_PLAYER, PLAYER1, PLAYER2, BoardPiece, PlayerAction
)

"""
This file tests the core game mechanics and board state logic for a Connect Four implementation.

There are 14 tests in total:
1. test_initialization: tests that the game board is correctly initialized with empty cells and proper shape.
2. test_conversion: tests that the board can be converted to a string and back without data loss.
3. test_valid_move: tests that valid column inputs are accepted as valid moves.
4. test_out_of_bounds: tests that out-of-range columns are rejected as invalid moves.
5. test_wrong_type_move: tests that non-integer move inputs (like strings or floats) are rejected.
6. test_full_column: tests that a completely filled column is rejected as a valid move.
7. test_apply_player_action_and_order: tests that pieces stack from bottom to top in a column.
8. test_connected_four_horizontal: tests detection of a horizontal winning sequence.
9. test_connected_four_vertical: tests detection of a vertical winning sequence.
10. test_connected_four_diagonal_down_right: tests detection of a diagonal win (bottom-left to top-right).
11. test_game_state_win: tests that the game detects a win condition correctly.
12. test_game_state_still_playing: tests that the game continues when no win condition is met.
13. test_connected_four_diagonal_down_left: tests detection of a diagonal win (top-right to bottom-left).
14. test_connected_four_no_win: tests that scattered pieces don't trigger a false positive win.
"""


def test_initialization():
    # 1. Now we will start by initializing the board.
    # 2. We want to test that the board returned is a NumPy array with the correct shape and type,
    #    and that all cells are initially set to NO_PLAYER (empty).
    board = initialize_game_state()
    assert isinstance(board, np.ndarray)  # Check that the board is a NumPy array
    assert board.shape == (6, 7)  # The board should have 6 rows and 7 columns
    assert board.dtype == BoardPiece  # Board values should be of type BoardPiece
    assert np.all(board == NO_PLAYER)  # Every position on the board should be empty

def test_conversion():
    # 1. Here we will test converting the board to a string and back again.
    # 2. We modify a few cells in the board and make sure that converting to a string
    #    and then back to a board gives us the same structure.
    board = initialize_game_state()
    board[5, 0] = PLAYER1  # Bottom-left cell
    board[5, 1] = PLAYER2  # Bottom row, second column
    board[4, 0] = PLAYER1  # One cell above the bottom-left

    pp = pretty_print_board(board)         # Convert board to string
    recovered = string_to_board(pp)        # Convert string back to board

    assert np.array_equal(board, recovered)  # Original and recovered boards must match

@pytest.mark.parametrize("col", [0, 3, 6])
def test_valid_move(col):
    # 1. We will test that valid columns are accepted.
    # 2. A move in any of these columns should be valid as the board is empty.
    board = initialize_game_state()
    assert check_move_status(board, PlayerAction(col)) == MoveStatus.IS_VALID
    

@pytest.mark.parametrize("col", [-1, 7, 100])
def test_out_of_bounds(col):
    # 1. We will test for column indices that are not within the allowed range.
    # 2. These should all be flagged as out-of-bounds.
    board = initialize_game_state()
    assert check_move_status(board, PlayerAction(col)) == MoveStatus.OUT_OF_BOUNDS

def test_wrong_type_move():
    # 1. Here we will test for wrong input types such as strings or floats.
    # 2. These should be rejected as valid moves.
    board = initialize_game_state()
    assert check_move_status(board, '3') == MoveStatus.WRONG_TYPE  # A string input
    assert check_move_status(board, 3.5) == MoveStatus.WRONG_TYPE  # A float input

def test_full_column():
    # 1. We will simulate filling an entire column.
    # 2. After filling it, any further move to that column should be rejected.
    board = initialize_game_state()
    col = 2
    for _ in range(6):
        apply_player_action(board, col, PLAYER1)  # Fill all 6 rows of column 2
    assert check_move_status(board, PlayerAction(col)) == MoveStatus.FULL_COLUMN

def test_apply_player_action_and_order():
    # 1. Here we will drop three pieces into the same column.
    # 2. We will then check that they appear in the correct order (stacked from bottom up).
    board = initialize_game_state()
    apply_player_action(board, 4, PLAYER1)  # Bottom row
    apply_player_action(board, 4, PLAYER2)  # One above the bottom
    apply_player_action(board, 4, PLAYER1)  # Two above the bottom

    assert board[0, 4] == PLAYER1 # bottom
    assert board[1, 4] == PLAYER2
    assert board[2, 4] == PLAYER1 # third from bottom

def test_connected_four_horizontal():
    # 1. We place four PLAYER1 pieces in a horizontal line on the bottom row.
    # 2. This should trigger a win detection.
    board = initialize_game_state()
    for col in range(4):
        apply_player_action(board, col, PLAYER1)
    assert connected_four(board, PLAYER1)

def test_connected_four_vertical():
    # 1. We drop four PLAYER2 pieces into the same column.
    # 2. This creates a vertical winning sequence.
    board = initialize_game_state()
    col = 0
    for _ in range(4):
        apply_player_action(board, col, PLAYER2)
    assert connected_four(board, PLAYER2)

def test_connected_four_diagonal_down_right():
    # 1. We create a diagonal from bottom-left to top-right for PLAYER1.
    # 2. This will test the diagonal win logic (down-right direction).
    board = initialize_game_state()
    moves = [
        (2, PLAYER1), 
        (3, PLAYER2), (3, PLAYER1),
        (4, PLAYER2), (4, PLAYER2), (4, PLAYER1),
        (5, PLAYER2), (5, PLAYER2), (5, PLAYER2), (5, PLAYER1)
    ]
    for col, player in moves:
        apply_player_action(board, col, player)
    assert connected_four(board, PLAYER1)

def test_game_state_win():
    # 1. We simulate a win condition by aligning 4 PLAYER1 pieces horizontally.
    # 2. Then we check that the game recognizes this as a win.
    board = initialize_game_state()
    for col in range(4):
        apply_player_action(board, col, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN

def test_game_state_still_playing():
    # 1. Place one piece and check that the game hasn't ended yet.
    board = initialize_game_state()
    apply_player_action(board, 0, PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING

def test_connected_four_diagonal_down_left():
    # 1. We create a diagonal from top-right to bottom-left for PLAYER2.
    # 2. This tests the other diagonal win detection logic.
    board = initialize_game_state()
    moves = [
        (3, PLAYER2), 
        (2, PLAYER1), (2, PLAYER2),
        (1, PLAYER1), (1, PLAYER1), (1, PLAYER2),
        (0, PLAYER1), (0, PLAYER1), (0, PLAYER1), (0, PLAYER2)
    ]
    for col, player in moves:
        apply_player_action(board, col, player)
    assert connected_four(board, PLAYER2)

def test_connected_four_no_win():
    # 1. Scatter pieces randomly without creating a win.
    # 2. Confirm that no false win is detected for either player.
    board = initialize_game_state()
    apply_player_action(board, 0, PLAYER1)
    apply_player_action(board, 1, PLAYER2)
    apply_player_action(board, 2, PLAYER1)
    apply_player_action(board, 3, PLAYER2)
    apply_player_action(board, 4, PLAYER1)
    apply_player_action(board, 5, PLAYER2)
    apply_player_action(board, 6, PLAYER1)

    assert not connected_four(board, PLAYER1)
    assert not connected_four(board, PLAYER2)
