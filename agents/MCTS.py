import numpy as np
import torch
import torch.nn.functional as F
from game_utils import *



def get_valid_actions(state):
    """Get list of valid actions (columns that aren't full)"""
    state = state.copy()
    valid_actions = []
    for col in range(7):
        try:
            apply_player_action(state, action=col, player=PLAYER1)
            valid_actions.append(col)
        except ValueError:
            # Column is full, skip it
            continue
    return valid_actions



def ucb_score(parent, child):
    """UCB score calculation with proper value perspective"""
    prior_score =   child.prior * np.sqrt(parent.visits) / (child.visits + 1)
    if child.visits == 0:
        return prior_score
    
    # Child value is from child's perspective, but we want it from parent's perspective
    # Since child represents the state after parent's move, we need to flip the sign
    value_score = -(child.value / child.visits)
    return value_score + prior_score

class Node:
    def __init__(self, prior, turn, state):
        self.prior = prior
        self.turn = turn
        next_player = PLAYER1 if turn == PLAYER2 else PLAYER2
        self.next_player = next_player
        self.state = state
        self.children = {}
        self.value = 0
        self.visits = 0

    def expand(self, action_probabilities):
        """
        Expands the node by creating child nodes for each possible action based on the action probabilities."""
        for action, prob in enumerate(action_probabilities):
            if prob > 0: 
                next_state = self.state.copy()
                try:
                    apply_player_action(next_state, action=action, player=self.turn)
                    self.children[action] = Node(prob, self.next_player, next_state)
                

                except ValueError:
                # invalid move → ingore this move
                    continue
        

    def select_child(self):
        """
        Selects a child node based on the UCT (Upper Confidence Bound for Trees) formula.
        """
        max_score = -99
        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > max_score:
                selected_action = action
                selected_child = child
                max_score = score
        return selected_action, selected_child





def MCTS_step(root, model_predict):
    """one MCTS step"""
    node = root
    search_path = [node]
    while len(node.children) > 0:
        _, node = node.select_child()
        search_path.append(node)
    

    # calculate value when leaf node is reached
#    terminal_state = check_end_state(board=node.state, player=node.turn)
#    if terminal_state == GameState.IS_WIN:
#        value = 1.0
    previous_player = PLAYER1 if node.turn == PLAYER2 else PLAYER2
    terminal_state = check_end_state(board=node.state, player=previous_player)
    if terminal_state == GameState.IS_WIN:
    # Previous player won, so current player (node.turn) lost
        value = -1.0
    elif terminal_state == GameState.IS_DRAW:
        value = 0.0
    else:
        value, action_probs = model_predict(node.state, node.turn)
        
        node.expand(action_probs)
    
      # Backpropagation phase
    for path_node in search_path:
        path_node.visits += 1
        # Value should be from the perspective of the player at this node
        if path_node.turn == search_path[-1].turn:
            path_node.value += value
        else:
            path_node.value += (-value)  # Flip value for opponent


def add_dirichlet_noise(root, alpha=0.35, epsilon=0.25):
    """
    Adds Dirichlet noise to the priors of the root node's children.
    
    Args:
        root (Node): root node of MCTS
        alpha (float): concentration parameter for Dirichlet distribution (typical: 0.3)
        epsilon (float): mixing factor between original prior and noise (typical: 0.25)
    """
    num_actions = len(root.children)
    noise = np.random.dirichlet([alpha] * num_actions)
    
    for i, (action, child) in enumerate(root.children.items()):
        # Mix original prior and noise
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]



def MCTS(root, model_predict, num_simulations=100, return_policy=False, add_noise=False):
    """
    Runs MCTS for a given number of simulations starting from the root node.
    """
    # Run MCTS simulations
    for i in range(num_simulations):
        MCTS_step(root, model_predict)
        
        # Add Dirichlet noise after first step (when root is expanded)
        if add_noise and i == 0 and len(root.children) > 0:
            add_dirichlet_noise(root, alpha=0.5, epsilon=0.35)
    
    # Calculate policy from visit counts
    visit_counts = np.zeros(7, dtype=np.float32)  # 7 columns in Connect 4
    for action, child in root.children.items():
        visit_counts[action] = child.visits
    
    if visit_counts.sum() == 0:
        raise RuntimeError("No MCTS visits recorded — check model or expansion logic")
    
    policy_target = visit_counts / visit_counts.sum()
    best_action = np.argmax(visit_counts)
    
    # Get root value (we can get this from the root node after simulations)
    root_value = root.value / root.visits if root.visits > 0 else 0.0
    
    if return_policy:
        return root_value, policy_target
    return best_action

def random_eval(model, device):
    random_values = np.random.rand(7)
    valuer = np.random.rand(1).item()  # Random value for the root node
    # Normalize so they sum to 1
    normalized_values = random_values / random_values.sum()

    # Convert to list if needed
    result = normalized_values.tolist()
    return valuer, result


def get_move_MCTS(board, player, saved_state=None, num_simulations=1000):
    """
    Generates a move using MCTS for the given board and player.
    """
    # Initialize the AlphaZero agent
    agent = AlphaZeroAgent()
    # Load your trained model
    agent.model.load_state_dict(torch.load("alphazero_connect4_final_500games_10iter_200depth.pth"))
   
    root = Node(prior=1.0, turn=player, state=board)
    best_action = MCTS(root, agent.predict, num_simulations=num_simulations)

    return PlayerAction(best_action), saved_state

def random_get_move_MCTS(board, player, saved_state=None, num_simulations=1000):
    """
    Generates a move using MCTS with random evaluation for the given board and player.
    """
   
    root = Node(prior=1.0, turn=player, state=board)
    best_action = MCTS(root, random_eval , num_simulations=num_simulations)

    return PlayerAction(best_action), saved_state


