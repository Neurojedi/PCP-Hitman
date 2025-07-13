# play_Alpha.py

"""
This script allows a human to play Connect Four against an AlphaZero-trained agent.

Usage examples from terminal:
    python play_game.py
        - You play as Player 1 (X) by default.

    python play_game.py --player 2
        - You play as Player 2 (O), Alpha goes first.

    python play_game.py --player 2 --model weights/checkpoint_iter_25.pth
        - Specify a custom trained model.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to enable absolute imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from neuralnet.ResNet import Connect4Model as ConnectFourNet
from agents.MCTS import Node, MCTS
from game_utils import (
    PLAYER1, PLAYER2, initialize_game_state, pretty_print_board,
    apply_player_action, check_end_state, GameState, PlayerAction
)
from human_user import user_move, SavedState
from neuralnet.utils import horizontal_flip, state_to_tensor, get_valid_actions, get_temperature


def load_model(path: str, device: str = "cpu") -> ConnectFourNet:
    """
    Load a trained ConnectFour model from the specified path.

    Args:
        path (str): Path to the model checkpoint file.
        device (str): Device to load the model on ("cpu" or "cuda").

    Returns:
        ConnectFourNet: The loaded PyTorch model.
    """
    checkpoint = torch.load(path, map_location=device)
    model = ConnectFourNet(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_model_predict_fn(model, device):
    """
    Wraps a model into a prediction function compatible with MCTS.

    Args:
        model: The PyTorch model for prediction.
        device (str): Device to run inference on.

    Returns:
        Callable: A function(state, turn) -> (value, policy_probs).
    """
    def predict(state, turn):
        model.eval()
        with torch.no_grad():
            # Convert game state to input tensor
            tensor = state_to_tensor(state, turn)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)

            # Run model prediction
            value, policy_logits = model(tensor)

            # Convert logits to probabilities
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()

            # Mask out invalid actions
            valid_actions = get_valid_actions(state)
            masked_probs = np.zeros(7)
            for action in valid_actions:
                masked_probs[action] = policy_probs[action]

            # Normalize masked probabilities
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
            else:
                # Fallback: uniform probability if softmax fails
                for action in valid_actions:
                    masked_probs[action] = 1.0 / len(valid_actions)

            return value.item(), masked_probs
    return predict


def play_against_alpha(model: ConnectFourNet, human_player=PLAYER1, device="cpu"):
    """
    Launch an interactive Connect Four game between a human and the AlphaZero agent.

    Args:
        model (ConnectFourNet): The trained AlphaZero model.
        human_player (int): PLAYER1 (1) or PLAYER2 (2), indicating which side the human plays.
        device (str): "cpu" or "cuda" for model inference.
    """
    board = initialize_game_state()
    current_player = PLAYER1
    saved_state = SavedState()
    model_predict = make_model_predict_fn(model, device)

    print("You are " + ("Player 1 (X)" if human_player == PLAYER1 else "Player 2 (O)"))

    while True:
        print(pretty_print_board(board))

        if current_player == human_player:
            # Human makes a move
            action, saved_state = user_move(board, current_player, saved_state)
        else:
            # Alpha agent makes a move using MCTS
            print("Alpha is thinking...")
            root = Node(prior=1.0, turn=current_player, state=board)
            best_action = MCTS(root, model_predict, num_simulations=100)
            action = best_action

        apply_player_action(board, action, current_player)

        # Check if game has ended
        state = check_end_state(board, current_player)
        if state != GameState.STILL_PLAYING:
            print(pretty_print_board(board))
            if state == GameState.IS_WIN:
                print("Player", "1 (You)" if current_player == human_player else "2 (Alpha)", "wins!")
            elif state == GameState.IS_DRAW:
                print("It's a draw!")
            break

        # Switch player turn
        current_player = PLAYER2 if current_player == PLAYER1 else PLAYER1


if __name__ == "__main__":
    # Parse command-line arguments

    # Example usage:
    #   python play_game.py --player 1 --model weights/checkpoint_iter_5.pth
    parser = argparse.ArgumentParser(description="Play Connect Four against AlphaZero.")
    parser.add_argument(
        "--player", type=int, choices=[1, 2], default=1,
        help="Choose your player: 1 (X, goes first) or 2 (O, goes second). Default is 1."
    )
    parser.add_argument(
        "--model", type=str, default="weights/checkpoint_iter_7.pth",
        help="Path to the trained model checkpoint."
    )
    args = parser.parse_args()

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and start game
    model = load_model(args.model, device=device)
    play_against_alpha(model, human_player=np.int8(args.player), device=device)
