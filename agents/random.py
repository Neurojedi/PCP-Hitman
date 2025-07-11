import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from game_utils import BoardPiece, PlayerAction, SavedState, MoveStatus, check_move_status

"""
Selects a random valid move for the given board state.

This function represents a non-strategic, random agent for Connect Four. It 
identifies all valid columns where a move can be made and randomly selects one.
If no valid moves are available (e.g., the board is full), it returns None.

Args:
    board (np.ndarray): The current game board state.
    current_player (BoardPiece): The player making the move.
    saved_state (SavedState, optional): State data preserved between turns. Defaults to None.

Returns:
    tuple[PlayerAction | None, SavedState | None]: A randomly chosen valid action 
    (column index), and the unchanged saved state. If no valid moves are available, 
    the action is None.
"""
def generate_move_random(board, current_player, saved_state=None):
    from neuralnet.utils import get_valid_actions

    valid_columns = get_valid_actions(board)
    # print(f"[Random Agent] Valid columns: {valid_columns}")  # DEBUG LINE

    if not valid_columns:
        print("[Random Agent] No valid moves. Returning None.")
        return None, saved_state

    action = np.random.choice(valid_columns)
    return action, saved_state
