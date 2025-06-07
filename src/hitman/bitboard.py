import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BitBoard():

    def _init__(self, p1_bb=0, p2_bb=0, turn=1):
        self.p1 = p1_bb
        self.p2 = p2_bb
        self.turn = turn

    def occupied(self):
        return self.p1 | self.p2
    
    def valid_moves(self):
        moves = []
        # we assume a board of shape 6,7 here TODO
        for column in range(7):
            # bit number 35 is the first bit of the top most row, we iterate over all bits in that row
            top_bit = 5*7 + column
            # we shift the current bit to position 0 and extract it
            if ((self.occupied() >> top_bit) & 1) == 0:
                # we add the column to the valid moves list
                moves.append(column)
        return moves
    
    def make_move(self, col):
        # we assume board size again here TODO
        # we iterate over all rows until we find an empty cell in the col or the col is full
        for row in range(6):
            bit = 1 << (row*7 + col)

            if (self.occupied & bit) == 0:
                if self.turn == 1:
                    # place the bit by using OR
                    new_p1 = self.p1 | bit
                    new_p2 = self.p2
                else:
                    new_p1 = self.p1
                    new_p2 = self.p2 | bit

                # 4) Flip the turn (-1 ↔ +1) and return a brand-new BitBoard
                return BitBoard(new_p1, new_p2, -self.turn)
            
        # if we do not find an empty cell TODO handle this in the loop
        raise ValueError(f"Column {col} is full")
    
    def is_win(self, player_bb):
        # Directions: horizontal (1), vertical (7), diag1 (6), diag2 (8)
        for shift in (1, 7, 6, 8):
            # we first find a mask with every two consecutive stones
            m = player_bb & (player_bb >> shift)
            # if we have two times two consecutive stones we have a win
            if m & (m >> (2 * shift)):
                return True
        return False

    def check_winner(self):
        if self.is_win(self.p1):
            return 1
        if self.is_win(self.p2):
            return -1
        if self.occupied() == (1 << 42) - 1:
            return 0
        return None
    
    def to_tensor(self, device=None):
        '''Convert bitboards to (1,3,6,7) FloatTensor for network input.'''
        mask = torch.arange(42, dtype=torch.int64, device=device)
        mask = (1 << mask)
        p1 = ((torch.tensor(self.p1, dtype=torch.int64, device=device) & mask) > 0).view(6,7).float()
        p2 = ((torch.tensor(self.p2, dtype=torch.int64, device=device) & mask) > 0).view(6,7).float()
        turn_plane = torch.full((6,7), float(self.turn), dtype=torch.float32, device=device)
        x = torch.stack([p1, p2, turn_plane], dim=0).unsqueeze(0)  # shape (1,3,6,7)
        return x