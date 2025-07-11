from typing import Callable, Optional, Any
from enum import Enum
import numpy as np


BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input does not have the correct type (PlayerAction).'
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).

    
    """
    return np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    Parameters
    ----------
    board : np.ndarray
        The game board to be printed, with shape (6, 7) and dtype BoardPiece.
    Returns
    -------
    str
        A string representation of the game board, formatted for human readability
    """
    symbol_map = {
        NO_PLAYER: NO_PLAYER_PRINT,
        PLAYER1: PLAYER1_PRINT,
        PLAYER2: PLAYER2_PRINT
    }
    lines = []

    lines.append("|" + "=" * (BOARD_COLS * 2) + "|")  # Top border

    for row in reversed(range(BOARD_ROWS)):  # Print from bottom to top
        row_pieces = [symbol_map[board[row, col]] for col in range(BOARD_COLS)]
        row_str = "|" + " ".join(row_pieces) + " |"
        lines.append(row_str)

    lines.append("|" + "=" * (BOARD_COLS *2) + "|")  # Bottom border

     # Add column indices line
    col_indices = "|" + " ".join(str(i) for i in range(BOARD_COLS)) + " |"
    lines.append(col_indices)

    return "\n".join(lines)


    
def string_to_board(pp_board: str) -> np.ndarray:
   """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    Parameters
    ----------
    pp_board : str
        The pretty printed board string representation.
    Returns
    -------
    np.ndarray
        The game board as a numpy array with shape (6, 7) and dtype BoardPiece.
    """
   board = initialize_game_state()
    
    # get the lines sepparately
    # and remove the first and last lines (the top and bottom borders)
   lines = pp_board.split('\n')
   board_lines = []
   for line in lines:
        stripped_line = line.strip()
        if (stripped_line.startswith('|') and
            stripped_line.endswith('|') and
            '=' not in stripped_line and
            '5' not in stripped_line):
            board_lines.append(stripped_line)


    # reverse to get the correct row order (from array row 0 to 5)
   board_lines = board_lines[::-1]
   if len(board_lines) != BOARD_ROWS:
        raise ValueError(f"Expected {BOARD_ROWS} board rows, found {len(board_lines)}")
        
   for row_idx, line in enumerate(board_lines):
        inner = line.strip('|')
        expected_length = 2 * BOARD_COLS 
        if len(inner) != expected_length:
            raise ValueError(f"Line length mismatch: expected {expected_length}, got {len(inner)}")
            
        cells = [inner[i] for i in range(0, expected_length, 2)]
        for col_idx, char in enumerate(cells):
            if char == PLAYER1_PRINT:
                board[row_idx, col_idx] = PLAYER1
            elif char == PLAYER2_PRINT:
                board[row_idx, col_idx] = PLAYER2
            else:
                board[row_idx, col_idx] = NO_PLAYER
   return board
                


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Sets board[i, action] = player, where i is the lowest open row. The input 
    board should be modified in place, such that it's not necessary to return 
    something.
    Parameters
    ----------
    board : np.ndarray
        The current game board.
    action : PlayerAction
        The column in which the player wants to place their piece.
    player : BoardPiece
        The player making the move, either PLAYER1 or PLAYER2.
    Raises
    ------
    ValueError
        If no open row is found in the specified column.
   
    """


    # find the lowest open row in the specified column
    for row in range(INDEX_LOWEST_ROW, BOARD_ROWS):
        if board[row, action] == NO_PLAYER:
            board[row, action] = player
            return
    # if no open row is found
    raise ValueError("No open row found in the specified column.")


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    Parameters
    ----------
    board : np.ndarray
        The current game board.
    player : BoardPiece
        The player whose pieces are to be checked for a connected four.
    Returns
    -------
    bool
        True if there is a connected four for the player, False otherwise.
    """
    #horizontal
    for row in range(BOARD_ROWS):
        for col in range(4):
            if (board[row, col] == player and
                board[row, col + 1] == player and
                board[row, col + 2] == player and
                board[row, col + 3] == player):
                return True
    #vertical
    for col in range(BOARD_COLS):
        for row in range(3):
            if (board[row, col] == player and
                board[row + 1, col] == player and
                board[row + 2, col] == player and
                board[row + 3, col] == player):
                return True
    # Check \ diagonals (top-left to bottom-right)
    for row in range(3):            # max row index: 2 → +3 = 5
        for col in range(4):        # max col index: 3 → +3 = 6
            if (board[row, col] == player and
                board[row + 1, col + 1] == player and
                board[row + 2, col + 2] == player and
                board[row + 3, col + 3] == player):
                return True

    # Check / diagonals (bottom-left to top-right)
    for row in range(3, 6):         # start at row 3, to allow -3 without underflow
        for col in range(4):        # max col index: 3 → +3 = 6
            if (board[row, col] == player and
                board[row - 1, col + 1] == player and
                board[row - 2, col + 2] == player and
                board[row - 3, col + 3] == player):
                return True



def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    Parameters
    ----------
    board : np.ndarray
        The current game board.
    player : BoardPiece
        The player whose last action is to be checked.
    Returns
    -------
    GameState
        The current game state, indicating if the game is won, drawn, or still ongoing.     
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    if np.all(board != NO_PLAYER):
        return GameState.IS_DRAW
    return GameState.STILL_PLAYING


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Returns a MoveStatus indicating whether a move is accepted as a valid move 
    or not, and if nwot, why.
    The provided column must be of the correct type (PlayerAction).
    Furthermore, the column must be within the bounds of the board and the
    column must not be full.

    Parameters
    ----------
    board : np.ndarray
        The current game board.
    column : Any
        The column to be played, which should be of type PlayerAction.
    Returns
    -------
    MoveStatus
        The status of the move, indicating if it is valid or why it is not. 
    """
    # is action valid?
    if not isinstance(column, PlayerAction):
        
        return MoveStatus.WRONG_TYPE
    
    # is action in bounds?
    
    if column < 0 or column >= BOARD_COLS:
        return MoveStatus.OUT_OF_BOUNDS
    
    if board[5, column] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN
    return MoveStatus.IS_VALID
    