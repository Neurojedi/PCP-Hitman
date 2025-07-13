"""
This script runs games between the AlphaZero model and a Minimax agent.
AlphaZero plays as Player 1 by default, but you can switch sides if needed.

Usage example:
    python play_AlphaMinimax.py \
      --model weights/checkpoint_iter_1.pth \
      --alpha_first 1 \
      --depth 6 \
      --heuristic window_count \
      --num_games 50
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# allow imports from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from neuralnet.ResNet import Connect4Model as ConnectFourNet
from agents.MCTS import Node, MCTS
from game_utils import (
    PLAYER1, PLAYER2,
    initialize_game_state,
    apply_player_action,
    check_end_state,
    GameState
)
from agents.minimax import generate_move_minimax
from neuralnet.utils import state_to_tensor, get_valid_actions


def load_model(path: str, device: str = "cpu") -> ConnectFourNet:
    """Load a trained AlphaZero model from the specified path."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ConnectFourNet(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_model_predict_fn(model, device):
    """Wraps the AlphaZero model to give (value, policy) for MCTS."""
    def predict(state, turn):
        model.eval()
        with torch.no_grad():
            tensor = state_to_tensor(state, turn)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)
            value, policy_logits = model(tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()

            valid_actions = get_valid_actions(state)
            masked = np.zeros_like(policy_probs)
            for a in valid_actions:
                masked[a] = policy_probs[a]

            if masked.sum() > 0:
                masked /= masked.sum()
            else:
                # fallback uniform
                for a in valid_actions:
                    masked[a] = 1.0 / len(valid_actions)
            return value.item(), masked
    return predict


def run_alpha_vs_minimax(
    model,
    alpha_first: bool = True,
    num_games: int = 100,
    device: str = "cpu",
    depth: int = 4,
    heuristic: str = "piece_diff"
):
    results = {"AlphaZero": 0, "Minimax": 0, "Draw": 0}
    model_predict = make_model_predict_fn(model, device)

    for i in range(1, num_games + 1):
        board = initialize_game_state()
        current_player = PLAYER1
        alpha_player = PLAYER1 if alpha_first else PLAYER2
        saved_mm = None  # for any bookkeeping in your minimax

        while True:
            if current_player == alpha_player:
                # AlphaZero move via MCTS
                root = Node(prior=1.0, turn=current_player, state=board)
                action = MCTS(root, model_predict, num_simulations=100)
            else:
                # Minimax move
                action, saved_mm = generate_move_minimax(
                    board, current_player, saved_mm,
                    depth=depth, heuristic=heuristic
                )

            apply_player_action(board, action, current_player)
            state = check_end_state(board, current_player)

            if state != GameState.STILL_PLAYING:
                print(f"Game {i}: ", end="")
                if state == GameState.IS_WIN:
                    winner = "AlphaZero" if current_player == alpha_player else "Minimax"
                    results[winner] += 1
                    print(f"{winner} wins!")
                else:
                    results["Draw"] += 1
                    print("It's a draw!")
                break

            current_player = PLAYER2 if current_player == PLAYER1 else PLAYER1

    # final tally
    print(f"\nSummary after {num_games} games:")
    for name, cnt in results.items():
        print(f"{name}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="weights/checkpoint_iter_7.pth",
        help="Path to the trained AlphaZero model"
    )
    parser.add_argument(
        "--alpha_first", type=int, choices=[0, 1], default=1,
        help="Whether AlphaZero plays first. 1 = Alpha, 0 = Minimax"
    )
    parser.add_argument(
        "--depth", type=int, default=16,
        help="Search depth for the Minimax agent"
    )
    parser.add_argument(
        "--heuristic", type=str,
        choices=["piece_diff", "center_weight", "window_count"],
        default="piece_diff",
        help="Board‐evaluation heuristic for Minimax"
    )
    parser.add_argument(
        "--num_games", type=int, default=1000,
        help="Number of games to play"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    run_alpha_vs_minimax(
        model,
        alpha_first=bool(args.alpha_first),
        num_games=args.num_games,
        device=device,
        depth=args.depth,
        heuristic=args.heuristic
    )
