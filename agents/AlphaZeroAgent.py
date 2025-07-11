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

def play_game(agent, num_simulations, temperature=1.0, return_metrics=False):
    """
    Plays a single self-play game using MCTS and an AlphaZero-style agent.

    Parameters:
    -----------
    agent : AlphaZeroAgent
        The agent providing value and policy predictions.
    
    num_simulations : int
        Number of MCTS simulations to run per move.
    
    temperature : float, optional
        Initial temperature to control exploration (default: 1.0). 
        In practice, this is overridden by a decay schedule.

    return_metrics : bool, optional
        If True, returns gameplay metrics like winner and game length.

    Returns:
    --------
    training_examples : list of (state, turn, policy, reward)
        Collected training data from self-play with outcome-based reward.

    metrics : dict, optional
        Returned only if `return_metrics=True`, includes 'game_length', 'winner', and 'is_draw'.
    """

    game_data = []  # Stores (state, turn, policy) for each move
    state = initialize_game_state()
    turn = PLAYER1
    move_count = 0

    while True:
        # Dynamically adjust temperature based on move count
        temperature = get_temperature(move_count)

        # Create the root node for MCTS using current game state
        root = Node(0, turn, state.copy())

        # Run MCTS to get policy (action probabilities)
        _, policy = MCTS(
            root, agent.predict, 
            num_simulations=num_simulations, 
            return_policy=True, 
            add_noise=True
        )

        # Store the current state, turn, and policy distribution
        game_data.append((state.copy(), turn, policy.copy()))

        # Sample an action from the policy distribution
        if temperature == 0:
            action = np.argmax(policy)  # Greedy
        else:
            # Apply temperature to encourage exploration
            policy_temp = np.power(policy, 1 / temperature)
            policy_temp /= policy_temp.sum()
            action = np.random.choice(7, p=policy_temp)

        # Check for game-over condition (board full)
        valid_actions = get_valid_actions(state)
        if valid_actions == []:
            break

        # If chosen action is invalid, sample a valid fallback
        if action not in valid_actions:
            print(f"Warning: chosen action {action} is invalid. Valid actions: {valid_actions}. Using fallback.")
            action = np.random.choice(valid_actions)

        # Apply the action to the board
        apply_player_action(state, action, turn)

        # Check if the game has ended
        game_state = check_end_state(state, turn)
        if game_state in (GameState.IS_WIN, GameState.IS_DRAW):
            break

        # Switch player turn
        turn = PLAYER2 if turn == PLAYER1 else PLAYER1
        move_count += 1

    # Assign final rewards to training examples
    if game_state == GameState.IS_WIN:
        winner = turn  # The last player who moved
    else:
        winner = None  # Draw

    training_examples = []
    for state, turn, policy in game_data:
        if winner is None:
            reward = 0
        elif winner == turn:
            reward = 1
        else:
            reward = -1
        training_examples.append((state, turn, policy, reward))

    # Optionally return game metrics
    if return_metrics:
        metrics = {
            'game_length': move_count + 1,
            'winner': winner,
            'is_draw': winner is None
        }
        return training_examples, metrics

    return training_examples

def evaluate_agent(agent1, agent2, num_games=100):
    """
    Evaluate agent1 against agent2 and a random agent using Monte Carlo Tree Search (MCTS).
    
    This function runs two sets of matches:
    1. agent1 vs. agent2.
    2. agent1 vs. a random evaluation agent.
    
    Parameters:
        agent1: The first agent with a 'predict' method and a 'model' attribute.
        agent2: The second agent with a 'predict' method and a 'model' attribute.
        num_games: Number of games to play in each evaluation set (default is 100).
    
    Returns:
        A tuple containing:
        - win_rate: agent1's win rate against agent2.
        - draw_rate: agent1's draw rate against agent2.
        - win_rate_random: agent1's win rate against the random evaluation agent.
        - draw_rate_random: agent1's draw rate against the random evaluation agent.
    """

    agent1.model.eval()
    agent2.model.eval()

    # Evaluate agent1 vs. agent2
    wins = 0
    draws = 0
    
    for game in range(num_games):
        
        state = initialize_game_state()  # Set up a new game
        turn = PLAYER1  # PLAYER1 is agent1

        while True:
            if turn == PLAYER1:
                # Use MCTS with agent1's prediction
                root = Node(0, turn, state.copy())
                action = MCTS(root, agent1.predict, num_simulations=100, return_policy=False, add_noise=True)
            else:
                # Use MCTS with agent2's prediction
                root = Node(0, turn, state.copy())
                action = MCTS(root, agent2.predict, num_simulations=100, return_policy=False, add_noise=True)

            # Ensure the action is valid; fallback to random valid action if not
            valid_actions = get_valid_actions(state)
            if action not in valid_actions:
                print(f"Warning: MCTS returned invalid action {action}. Valid actions: {valid_actions}.")
                if valid_actions:
                    action = np.random.choice(valid_actions)
                else:
                    # No valid moves left
                    break
            
            apply_player_action(state, action, turn)  # Apply chosen action to the game state
            game_state = check_end_state(state, turn)  # Check for win/draw/end

            if game_state == GameState.IS_WIN:
                if turn == PLAYER1:
                    wins += 1
                break
            elif game_state == GameState.IS_DRAW:
                draws += 1
                break
            
            # Switch turns
            turn = PLAYER2 if turn == PLAYER1 else PLAYER1

    win_rate = wins / num_games
    draw_rate = draws / num_games

    # Evaluate agent1 vs. random_eval agent
    wins_random = 0
    draws_random = 0
    
    for game in range(num_games):
        state = initialize_game_state()
        turn = PLAYER1

        while True:
            if turn == PLAYER1:
                # agent1's turn using MCTS with its prediction model
                root = Node(0, turn, state.copy())
                action = MCTS(root, agent1.predict, num_simulations=100, return_policy=False)
            else:
                # Random evaluation agent's turn using MCTS with random evaluation
                root = Node(0, turn, state.copy())
                action = MCTS(root, random_eval, num_simulations=100, return_policy=False)

            valid_actions = get_valid_actions(state)
            if action not in valid_actions:
                print(f"Warning: MCTS returned invalid action {action}. Valid actions: {valid_actions}.")
                if valid_actions:
                    action = np.random.choice(valid_actions)
                else:
                    break
            
            apply_player_action(state, action, turn)
            game_state = check_end_state(state, turn)

            if game_state == GameState.IS_WIN:
                if turn == PLAYER1:
                    wins_random += 1
                break
            elif game_state == GameState.IS_DRAW:
                draws_random += 1
                break
            
            turn = PLAYER2 if turn == PLAYER1 else PLAYER1

    win_rate_random = wins_random / num_games
    draw_rate_random = draws_random / num_games

    return win_rate, draw_rate, win_rate_random, draw_rate_random


class AlphaZeroAgent:
    """
    AlphaZero-style agent for Connect4 using a neural network to predict
    value and policy from board states.

    Attributes:
    -----------
    device : str
        The computation device ('cuda' or 'cpu').

    model : Connect4Model
        The neural network model used for prediction.

    optimizer : torch.optim.Optimizer
        Optimizer for training the model (Adam).

    scheduler : torch.optim.lr_scheduler.StepLR
        Learning rate scheduler to decay LR periodically.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize the neural network model
        self.model = Connect4Model(device).to(device)

        # Adam optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)

        # Learning rate scheduler to reduce LR every few epochs
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,  # Reduce LR every 3 steps
            gamma=0.8     # Multiply LR by 0.8 each time
        )

        # Enable CUDA benchmarking for faster convolution performance
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True

    def predict(self, state, turn):
        """
        Perform inference to get value and policy predictions for a given state.

        Parameters:
        -----------
        state : np.ndarray
            The current Connect4 board.

        turn : int
            The current player's turn (PLAYER1 or PLAYER2).

        Returns:
        --------
        value : float
            Predicted value of the current board position.

        masked_probs : np.ndarray of shape (7,)
            Probability distribution over valid actions (softmax over policy logits).
        """
        self.model.eval()
        with torch.no_grad():
            # Convert state to input tensor
            tensor = state_to_tensor(state, turn)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

            # Run model
            value, policy_logits = self.model(tensor)

            # Convert logits to probabilities using softmax
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()

            # Mask invalid actions
            valid_actions = get_valid_actions(state)
            masked_probs = np.zeros(7)
            for action in valid_actions:
                masked_probs[action] = policy_probs[action]

            # Renormalize masked probabilities
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # Fallback to uniform distribution if all masked
                for action in valid_actions:
                    masked_probs[action] = 1.0 / len(valid_actions)

            return value.item(), masked_probs

    def train_step(self, states, policy_targets, value_targets):
        """
        Perform a single training step on a batch of data.

        Parameters:
        -----------
        states : list of (state, turn)
            Each element is a board state and the player to move.

        policy_targets : np.ndarray of shape (B, 7)
            Ground truth policy distributions from MCTS.

        value_targets : np.ndarray of shape (B,)
            Ground truth value estimates.

        Returns:
        --------
        total_loss : float
            Total combined loss (weighted sum of value and policy loss).

        value_loss : float
            Mean squared error between predicted and target values.

        policy_loss : float
            KL divergence between predicted and target policy distributions.
        """
        self.model.train()

        # Convert batch of states to tensor input
        state_tensors = []
        for state, turn in states:
            tensor = state_to_tensor(state, turn)
            state_tensors.append(tensor)
        state_batch = torch.from_numpy(np.array(state_tensors)).to(self.device)

        # Convert target policies and values to tensors
        policy_batch = torch.from_numpy(np.array(policy_targets)).to(self.device)
        value_batch = torch.from_numpy(np.array(value_targets)).float().to(self.device)

        # Forward pass through the model
        pred_values, pred_policy_logits = self.model(state_batch)

        # Compute value loss (regression)
        value_loss = F.mse_loss(pred_values.squeeze(), value_batch)

        # Compute policy loss (KL divergence between softmaxed logits and target)
        log_probs = F.log_softmax(pred_policy_logits, dim=1)
        policy_loss = F.kl_div(log_probs, policy_batch, reduction='batchmean')

        # Combine losses (60% policy, 40% value)
        total_loss = 0.6 * policy_loss + 0.4 * value_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()

        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        return total_loss.item(), value_loss.item(), policy_loss.item()
