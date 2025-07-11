import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from game_utils import (
    initialize_game_state, 
    check_end_state, 
    GameState, 
    apply_player_action, 
    PLAYER1, 
    PLAYER2
)
from neuralnet.ResNet import Connect4Model
#python -m agents.alpha_zero_agent.Neural_Network_alternative

def horizontal_flip(state, policy):
    """
    Perform a horizontal flip on both the game state and the associated policy.

    This is typically used for data augmentation in board games with symmetrical properties,
    such as Connect Four.

    Parameters:
        state (np.ndarray): The current game state. 
            - 2D array: shape (height, width) — flips along columns.
            - 3D array: shape (channels, height, width) — flips along width dimension.
        policy (np.ndarray): A 1D array representing action probabilities or logits 
            corresponding to each column/action (e.g., length = number of columns).

    Returns:
        flipped_state (np.ndarray): The horizontally flipped game state.
        flipped_policy (np.ndarray): The horizontally flipped policy vector.

    Raises:
        ValueError: If state does not have 2 or 3 dimensions.
    """
    # If state is 2D (e.g., [6, 7] board), flip along axis 1 (columns)
    if state.ndim == 2:
        flipped_state = np.flip(state, axis=1).copy()
    # If state is 3D (e.g., [channels, height, width]), flip along axis 2 (columns)
    elif state.ndim == 3:
        flipped_state = np.flip(state, axis=2).copy()
    else:
        # Raise an error for unexpected input dimensions
        raise ValueError("Unexpected state dimensions: {}".format(state.ndim))

    # Flip the policy vector horizontally (reverse its elements)
    flipped_policy = np.flip(policy).copy()
    
    return flipped_state, flipped_policy


def state_to_tensor(state, turn):
    """
    Converts the Connect4 game board state into a 3-channel tensor suitable for input to a neural network.

    Parameters:
    -----------
    state : np.ndarray of shape (6, 7)
        The current game board, where each cell contains 0 (empty), 1 (Player 1), or 2 (Player 2).
    
    turn : int
        The player whose turn it is to play. Should be either PLAYER1 or PLAYER2.

    Returns:
    --------
    tensor : np.ndarray of shape (3, 6, 7)
        A binary encoded tensor representing the game state:
        - Channel 0: Binary mask of the current player's pieces.
        - Channel 1: Binary mask of the opponent's pieces.
        - Channel 2: All ones (can be used to help the network identify valid board shape or bias).
    """
    # Initialize an empty tensor with 3 channels, matching the board shape
    tensor = np.zeros((3, 6, 7), dtype=np.float32)

    # Identify the current player and the opponent
    current_player = turn
    opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1

    # Channel 0: positions of current player's pieces
    tensor[0] = (state == current_player).astype(np.float32)

    # Channel 1: positions of opponent's pieces
    tensor[1] = (state == opponent).astype(np.float32)

    # Channel 2: filled with ones to provide a constant feature
    tensor[2] = np.ones((6, 7), dtype=np.float32)

    return tensor

def get_valid_actions(state):
    """
    Returns a list of valid columns where a new piece can be dropped in the Connect4 game.

    Parameters:
    -----------
    state : np.ndarray of shape (6, 7)
        The current game board. Each cell contains 0 (empty), 1 (Player 1), or 2 (Player 2).

    Returns:
    --------
    valid_actions : list of int
        A list of column indices (0 to 6) that are not full and thus valid for placing a new piece.
    """
    # A column is valid if the top cell (row 5) is still empty
    return [col for col in range(7) if state[5, col] == 0]

# Temperature schedule for controlling exploration during MCTS
def get_temperature(move_count, total_moves=42):
    """
    Returns a temperature value based on the current move count to control exploration.

    The temperature gradually decreases as the game progresses:
    - Early game: high temperature (more exploration)
    - Mid game: moderate exploration
    - Late game: low to no exploration (greedy choice)

    Parameters:
    -----------
    move_count : int
        The current move number in the game (starting from 0).

    total_moves : int, default=42
        Total number of possible moves in a Connect4 game (6 rows × 7 columns).

    Returns:
    --------
    temperature : float
        Temperature parameter for softmax sampling.
    """
    if move_count < total_moves * 0.5:       # Early game: encourage exploration
        return 1.2
    elif move_count < total_moves * 0.7:     # Mid game: moderate exploration
        return 0.6
    elif move_count < total_moves * 0.9:     # Late game: minimal exploration
        return 0.1
    else:                                    # Final moves: deterministic (argmax)
        return 0.0
