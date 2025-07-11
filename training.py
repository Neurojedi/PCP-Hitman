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
from neuralnet.utils import horizontal_flip, state_to_tensor, get_valid_actions, get_temperature
from neuralnet.ResNet import Connect4Model
from agents.MCTS import Node, MCTS, random_eval  # Imported here in case of dynamic module context
from agents.AlphaZeroAgent import play_game, evaluate_agent, AlphaZeroAgent
def train_alphazero():
    """
    Main training loop for AlphaZero-style agent.

    This function performs iterative training using self-play, training updates, 
    and evaluation against a reference (best) agent. It also saves checkpoints 
    and tracks relevant metrics such as win rates and training losses.

    Workflow:
    1. Self-play to generate training data.
    2. Data augmentation using horizontal flips.
    3. Training neural network on accumulated data.
    4. Evaluation against the best agent and random agent.
    5. Model checkpointing.

    Hyperparameters and behaviors are hardcoded within the function.
    """

    # Hyperparameters
    ITERATIONS = 10  # Number of full training iterations
    SELF_PLAY_GAMES = 50  # Games generated per iteration via self-play
    MCTS_SIMULATIONS = 200  # Number of MCTS simulations per move
    BATCH_SIZE = 1024  # Batch size for neural network training
    BUFFER_SIZE = 150000  # Maximum size of training data buffer
    EVALUATION_GAMES = 100  # Number of evaluation games per iteration
    CHECKPOINT_INTERVAL = 1  # Frequency of checkpoint saving (in iterations)
    MAX_BATCHES_PER_ITER = 2000  # Cap on the number of batches per training iteration
    SIMULATION_INCREASE = 25  # Increment of MCTS simulations
    SIMULATION_INCREASE_Increment = 2  # Iteration interval to increase MCTS simulations

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize current agent and best agent (used for evaluation and as a baseline)
    current_agent = AlphaZeroAgent(device)
    best_agent = AlphaZeroAgent(device)
    best_agent.model.load_state_dict(current_agent.model.state_dict())

    # Initialize replay buffer and history tracking
    training_buffer = deque(maxlen=BUFFER_SIZE)
    game_metrics_history = []
    game_loss_metrics_history = []
    evaluation_history = []
    loss_metrics_history = []

    for iteration in range(ITERATIONS):
        # Optionally increase MCTS simulations
        if iteration % SIMULATION_INCREASE_Increment == 0 and iteration > 0 and MCTS_SIMULATIONS < 301:
            MCTS_SIMULATIONS += SIMULATION_INCREASE
            print(f"Using {MCTS_SIMULATIONS} MCTS simulations per move")

        print(f"\n=== Iteration {iteration + 1}/{ITERATIONS} ===")
        current_lr = current_agent.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.7f}")

        print("Self-play...")

        iteration_data = []
        iteration_metrics = []
        iteration_losses = []

        # Generate self-play games
        for game in range(SELF_PLAY_GAMES):
            game_data, game_metrics = play_game(current_agent, num_simulations=MCTS_SIMULATIONS, return_metrics=True)

            # Data augmentation: add original and horizontally flipped states
            augmented_game_data = []
            for state, turn, policy, reward in game_data:
                augmented_game_data.append((state, turn, policy, reward))
                flipped_state, flipped_policy = horizontal_flip(state, policy)
                augmented_game_data.append((flipped_state, turn, flipped_policy, reward))

            iteration_data.extend(augmented_game_data)
            iteration_metrics.append(game_metrics)

            # Optional immediate training per game
            states = [(data[0], data[1]) for data in game_data]
            policies = [data[2] for data in game_data]
            values = [data[3] for data in game_data]

            loss, value_loss, policy_loss = current_agent.train_step(states, policies, values)

            iteration_losses.append({
                'loss': loss,
                'value_loss': value_loss,
                'policy_loss': policy_loss
            })

            print(f"  Game {game + 1}/{SELF_PLAY_GAMES} completed. Metrics: {game_metrics}, Losses: {iteration_losses[-1]}")

        training_buffer.extend(iteration_data)
        game_metrics_history.append(iteration_metrics)
        game_loss_metrics_history.append(iteration_losses)

        print(f"Training buffer size: {len(training_buffer)}")

        avg_loss = avg_value_loss = avg_policy_loss = None

        # Train neural network using replay buffer if enough data accumulated
        if len(training_buffer) >= BATCH_SIZE:
            print("Training neural network...")
            shuffled_data = list(training_buffer)
            random.shuffle(shuffled_data)

            num_batches = len(shuffled_data) // BATCH_SIZE
            num_batches = min(num_batches, MAX_BATCHES_PER_ITER)

            total_loss = 0.0
            total_value_loss = 0.0
            total_policy_loss = 0.0

            for batch_idx in range(num_batches):
                start = batch_idx * BATCH_SIZE
                end = (batch_idx + 1) * BATCH_SIZE
                batch = shuffled_data[start:end]

                states = [(data[0], data[1]) for data in batch]
                policies = [data[2] for data in batch]
                values = [data[3] for data in batch]

                loss, value_loss, policy_loss = current_agent.train_step(states, policies, values)
                total_loss += loss
                total_value_loss += value_loss
                total_policy_loss += policy_loss

            # Average losses for reporting
            avg_loss = total_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches
            print(f"Avg Loss={avg_loss:.4f}, Value={avg_value_loss:.4f}, Policy={avg_policy_loss:.4f}")

        # Record loss metrics for the iteration
        loss_metrics_history.append({
            'iteration': iteration + 1,
            'avg_loss': avg_loss,
            'avg_value_loss': avg_value_loss,
            'avg_policy_loss': avg_policy_loss
        })

        # Update learning rate using scheduler
        current_agent.scheduler.step()
        new_lr = current_agent.optimizer.param_groups[0]['lr']
        print(f"Updated learning rate: {new_lr:.7f}")

        # Evaluate current agent against the best agent and random agent
        print("Evaluating against best agent...")
        win_rate, draw_rate, win_rate_random, draw_rate_random = evaluate_agent(
            current_agent, best_agent, EVALUATION_GAMES
        )
        print(f"Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}, "
              f"Win rate random: {win_rate_random:.2f}, Draw rate random: {draw_rate_random:.2f}")

        evaluation_history.append({
            'iteration': iteration + 1,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'win_rate_random': win_rate_random,
            'draw_rate_random': draw_rate_random
        })

        # Update best agent if current agent outperforms
        if win_rate > (1 - win_rate - draw_rate):
            best_agent.model.load_state_dict(current_agent.model.state_dict())
            print(f"New best agent! Win rate: {win_rate:.2f}")
        else:
            current_agent.model.load_state_dict(best_agent.model.state_dict())
            print(f"Reverting to best agent. Win rate was only {win_rate:.2f}")

        # Save checkpoint at specified intervals
        if (iteration + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"weights/checkpoint_iter_{iteration + 1}.pth"
            torch.save({
                'model_state_dict': current_agent.model.state_dict(),
                'optimizer_state_dict': current_agent.optimizer.state_dict(),
                'iteration': iteration,
                'metrics': game_metrics_history,
                'game_loss_metrics': game_loss_metrics_history,
                'evaluation': evaluation_history,
                'losses': loss_metrics_history,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model state after all iterations complete
    torch.save(current_agent.model.state_dict(), "weights/alphazero_connect4_final.pth")
    print("Training completed!")



if __name__ == "__main__":
    train_alphazero()