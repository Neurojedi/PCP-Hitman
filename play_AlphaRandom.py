"""
This script runs 10 games between the AlphaZero model and a random agent.
AlphaZero plays as Player 1 by default, but you can switch sides if needed.

Usage example:
    python play_AlphaRandom.py --model weights/checkpoint_iter_1.pth --alpha_first 1
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from neuralnet.ResNet import Connect4Model as ConnectFourNet
from agents.MCTS import Node, MCTS
from game_utils import (
    PLAYER1, PLAYER2, initialize_game_state, apply_player_action, check_end_state, GameState
)
from agents.random import generate_move_random
from neuralnet.utils import state_to_tensor, get_valid_actions


def load_model(path: str, device: str = "cpu") -> ConnectFourNet:
    """Load a trained model from the specified path."""
    checkpoint = torch.load(path, map_location=device)
    model = ConnectFourNet(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_model_predict_fn(model, device):
    """Wraps model to be used by MCTS."""
    def predict(state, turn):
        model.eval()
        with torch.no_grad():
            tensor = state_to_tensor(state, turn)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)
            value, policy_logits = model(tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()

            valid_actions = get_valid_actions(state)
            masked_probs = np.zeros(7)
            for action in valid_actions:
                masked_probs[action] = policy_probs[action]

            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
            else:
                for action in valid_actions:
                    masked_probs[action] = 1.0 / len(valid_actions)

            return value.item(), masked_probs
    return predict


def run_alpha_vs_random(model, alpha_first=True, num_games=1000, device="cpu"):
    results = {"AlphaZero": 0, "Random": 0, "Draw": 0}
    model_predict = make_model_predict_fn(model, device)

    for i in range(num_games):
        board = initialize_game_state()
        current_player = PLAYER1
        alpha_player = PLAYER1 if alpha_first else PLAYER2
        saved_state = None

        while True:
            if current_player == alpha_player:
                root = Node(prior=1.0, turn=current_player, state=board)
                action = MCTS(root, model_predict, num_simulations=100)
            else:
                valid_moves = get_valid_actions(board)
                print(f"Game {i+1} - Current player: {current_player}, valid moves: {valid_moves}")  # Debug print

                if not valid_moves:
                    print("No valid moves for Random agent - ending game as draw.")
                    state = GameState.IS_DRAW
                    break

                action, saved_state = generate_move_random(board, current_player, saved_state)

            apply_player_action(board, action, current_player)
            state = check_end_state(board, current_player)

            if state != GameState.STILL_PLAYING:
                print(f"Game {i+1}: ", end="")
                if state == GameState.IS_WIN:
                    winner = "AlphaZero" if current_player == alpha_player else "Random"
                    results[winner] += 1
                    print(f"{winner} wins!")
                elif state == GameState.IS_DRAW:
                    results["Draw"] += 1
                    print("It's a draw!")
                break

            current_player = PLAYER2 if current_player == PLAYER1 else PLAYER1

    print("\nSummary after", num_games, "games:")
    for key, value in results.items():
        print(f"{key}: {value}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="weights/checkpoint_iter_7.pth",
                        help="Path to the trained AlphaZero model")
    parser.add_argument("--alpha_first", type=int, choices=[0, 1], default=1,
                        help="Whether AlphaZero plays first. 1 = Alpha, 0 = Random")

    # Example usage:
    # python play_AlphaRandom.py --model weights/checkpoint_iter_1.pth --alpha_first 0
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    run_alpha_vs_random(model, alpha_first=bool(args.alpha_first), device=device)
