import numpy as np
from typing import Optional, Tuple
from game_utils import (
    NO_PLAYER,
    PLAYER1,
    PLAYER2,
    PlayerAction,
    SavedState,
    apply_player_action,
    check_end_state,
    GameState
)

def evaluate_board(
    board: np.ndarray,
    player: np.int8,
    heuristic: str = "piece_diff"
) -> float:
    """
    Evaluate a Connect-4 board state from the perspective of a given player.

    Implements three heuristic strategies:

    1. piece_diff:
       Computes (# of player's pieces) - (# of opponent's pieces).
    2. center_weight:
       Gives double weight to pieces in the central column.
    3. window_count:
       Scans all contiguous windows of length four,
       assigns scores for two- and three-in-a-row,
       and returns a large positive/negative value for a winning window.

    Args:
        board: A 2D NumPy array representing the game board.
        player: The identifier for the current player (PLAYER1 or PLAYER2).
        heuristic: The name of the heuristic to use.

    Returns:
        A float score indicating board favorability. Higher is better for `player`.
    """
    # Determine opponent
    opponent = PLAYER1 if player == PLAYER2 else PLAYER2
    rows, cols = board.shape

    if heuristic == "piece_diff":
        return float((board == player).sum() - (board == opponent).sum())

    if heuristic == "center_weight":
        center_col = cols // 2
        player_count = (board[:, center_col] == player).sum()
        opponent_count = (board[:, center_col] == opponent).sum()
        return float(player_count * 2 - opponent_count * 2)

    if heuristic == "window_count":
        def score_window(window: np.ndarray) -> float:
            p_count = int((window == player).sum())
            o_count = int((window == opponent).sum())
            if p_count == 4:
                return 1e6
            if o_count == 4:
                return -1e6
            if p_count == 3 and o_count == 0:
                return 5.0
            if p_count == 2 and o_count == 0:
                return 2.0
            if o_count == 3 and p_count == 0:
                return -4.0
            return 0.0

        score = 0.0
        # Horizontal windows
        for r in range(rows):
            for c in range(cols - 3):
                window = board[r, c : c + 4]
                score += score_window(window)
        # Vertical windows
        for c in range(cols):
            for r in range(rows - 3):
                window = board[r : r + 4, c]
                score += score_window(window)
        # Diagonal down-right
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r + i, c + i] for i in range(4)])
                score += score_window(window)
        # Diagonal up-right
        for r in range(3, rows):
            for c in range(cols - 3):
                window = np.array([board[r - i, c + i] for i in range(4)])
                score += score_window(window)

        return score

    # Unrecognized heuristic: default to neutral
    return 0.0


def minimax(
    board: np.ndarray,
    player: np.int8,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    heuristic: str
) -> Tuple[float, Optional[int]]:
    """
    Perform α–β pruned minimax search to select best move.

    Args:
        board: Current game board as a 2D NumPy array.
        player: The maximizing player's identifier.
        depth: Maximum search depth (remaining plies).
        alpha: Current α value for alpha-beta pruning.
        beta: Current β value for alpha-beta pruning.
        maximizing: True to maximize player's score, False to minimize.
        heuristic: Heuristic name passed to evaluate_board.

    Returns:
        A tuple of (value, best_column), where `value` is the minimax score,
        and `best_column` is the column index of the best move (or None if terminal).
    """
    opponent = PLAYER1 if player == PLAYER2 else PLAYER2

    # Terminal state checks
    state = check_end_state(board, player)
    if state == GameState.IS_WIN:
        return float("inf"), None
    state = check_end_state(board, opponent)
    if state == GameState.IS_WIN:
        return float("-inf"), None
    state = check_end_state(board, player)
    if state == GameState.IS_DRAW:
        return 0.0, None
    # Depth cutoff
    if depth == 0:
        return evaluate_board(board, player, heuristic), None

    # List of non-full columns
    valid_columns = [c for c in range(board.shape[1]) if board[0, c] == NO_PLAYER]
    best_col: Optional[int] = None

    if maximizing:
        max_val = float("-inf")
        for col in valid_columns:
            new_board = board.copy()
            apply_player_action(new_board, PlayerAction(col), player)
            val, _ = minimax(
                new_board, player, depth - 1, alpha, beta, False, heuristic
            )
            if val > max_val:
                max_val, best_col = val, col
            alpha = max(alpha, max_val)
            if alpha >= beta:
                break
        return max_val, best_col
    else:
        min_val = float("inf")
        for col in valid_columns:
            new_board = board.copy()
            apply_player_action(new_board, PlayerAction(col), opponent)
            val, _ = minimax(
                new_board, player, depth - 1, alpha, beta, True, heuristic
            )
            if val < min_val:
                min_val, best_col = val, col
            beta = min(beta, min_val)
            if beta <= alpha:
                break
        return min_val, best_col


def generate_move_minimax(
    board: np.ndarray,
    player: np.int8,
    saved_state: Optional[SavedState],
    depth: int = 4,
    heuristic: str = "piece_diff"
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Entry point for a minimax-based Connect-4 agent.

    Picks the best column via minimax search, with random fallback.

    Args:
        board: Current game board as a 2D NumPy array.
        player: Identifier of the acting player.
        saved_state: Unused for minimax; retained for interface compatibility.
        depth: Maximum recursion depth for search.
        heuristic: Name of heuristic to evaluate non-terminal nodes.

    Returns:
        A tuple of (PlayerAction, saved_state). `saved_state` is unchanged.
    """
    score, selected_col = minimax(
        board,
        player,
        depth,
        float("-inf"),
        float("inf"),
        True,
        heuristic,
    )

    if selected_col is None:
        # Fall back: choose a random valid move
        valid_cols = [c for c in range(board.shape[1]) if board[0, c] == NO_PLAYER]
        selected_col = int(np.random.choice(valid_cols))

    return PlayerAction(selected_col), saved_state
