import itertools
from games.Game import Game


class Connect4(Game):
    def __init__(self, agents, state=None, *args, **kwargs):
        assert len(agents) == 2, f'There should be 2 agents, but there are {len(agents)}.'
        if state is None:
            self.state = [[' ', ' ', ' ', ' ', ' ', ' ', ' '], # ' ' => empty
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '], # 0 => agent 0 played there
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                          [' ', ' ', ' ', ' ', ' ', ' ', ' ']] # 1 => agent 1 played there
        else:
            self.state = state
        super().__init__(self.state, agents, *args, **kwargs)

    def result(self):
        """
        Detmine the winner of the game:
            Return agent index (i.e. `turn`) of winning player if a player has won.
            Return None the game is not over.
            Return -1 if the game is over and there is no winner (i.e. a tie).
        """
        # check for winner
        # assume the top move in one of the columns is part of the connect 4 (if it exists)
        for col in range(7):
            # set row to be the index of the top move in column col (TODO: could optimize by starting from bottom)
            row = 0
            while row < 6 and self.state[row][col] == ' ':
                row += 1
            if row == 6:
                continue
            val = self.state[row][col]
            ### check for 4 in a row through (row, col)
            # vertical case
            if row < 3:
                if val == self.state[row+1][col] == self.state[row+2][col] == self.state[row+3][col]:
                    return val
            # horizontal case
            cnt = 1
            if col > 0: # moving left
                j = col - 1
                while (0 <= j and self.state[row][j] == val):
                    cnt += 1
                    j -= 1
            if col < 6: # moving right
                j = col + 1
                while (j < 7 and self.state[row][j] == val):
                    cnt += 1
                    j += 1	
            if cnt >= 4:
                return val
            # diagonal down-left/up-right case
            cnt = 1
            if col > 0 and row < 5: # moving down-left
                i = row + 1
                j = col - 1
                while (i < 6 and 0 <= j and self.state[i][j] == val):
                    cnt += 1
                    i += 1
                    j -= 1
            if col < 6 and row > 0: # moving up-right
                i = row - 1
                j = col + 1
                while (0 <= i and j < 7 and self.state[i][j] == val):
                    cnt += 1
                    i -= 1
                    j += 1
            if cnt >= 4:
                return val
            # diagonal up-left/down-right case
            cnt = 1
            if row > 0 and col > 0: # moving up-left
                i = row - 1
                j = col - 1
                while (0 <= i and 0 <= j and self.state[i][j] == val):
                    cnt += 1
                    i -= 1
                    j -= 1
            if row < 5 and col < 6: # moving down-right
                i = row + 1
                j = col + 1
                while (i < 6 and j < 7 and self.state[i][j] == val):
                    cnt += 1
                    i += 1
                    j += 1
            if cnt >= 4:
                return val
        # if the game is not over, return None
        for col in range(7):
            if self.state[0][col] == ' ':
                return None
        # if it's a tie, return -1
        return -1

    def pretty_print(self):
        """print game state in human-readable format."""
        print("\n  1   2   3   4   5   6   7  ")
        print(f"| {self.state[0][0]} | {self.state[0][1]} | {self.state[0][2]} | {self.state[0][3]} | {self.state[0][4]} | {self.state[0][5]} | {self.state[0][6]} |")
        print(f"| {self.state[1][0]} | {self.state[1][1]} | {self.state[1][2]} | {self.state[1][3]} | {self.state[1][4]} | {self.state[1][5]} | {self.state[1][6]} |")
        print(f"| {self.state[2][0]} | {self.state[2][1]} | {self.state[2][2]} | {self.state[2][3]} | {self.state[2][4]} | {self.state[2][5]} | {self.state[2][6]} |")
        print(f"| {self.state[3][0]} | {self.state[3][1]} | {self.state[3][2]} | {self.state[3][3]} | {self.state[3][4]} | {self.state[3][5]} | {self.state[3][6]} |")
        print(f"| {self.state[4][0]} | {self.state[4][1]} | {self.state[4][2]} | {self.state[4][3]} | {self.state[4][4]} | {self.state[4][5]} | {self.state[4][6]} |")
        print(f"| {self.state[5][0]} | {self.state[5][1]} | {self.state[5][2]} | {self.state[5][3]} | {self.state[5][4]} | {self.state[5][5]} | {self.state[5][6]} |")

    @staticmethod
    def move_index(move):
        """get index of move from tuple"""
        return move[1]    

    def get_data_point(self, process=True, **kwargs):
        """
        grab a random game state, and output the tuple (state, turn, move, winner)
            `state` is the current game state
            `turn` is the index of the agent that plays next
            `move` is the move the next player will make
            `winner` is the index of the agent that won, or -1 if the game was a tie.
            `process` = True => replace " "s, and ordinal encode move, so these can be used as input to NN training
        this will be used to generate training data.
        """
        state, turn, move, winner = super().get_data_point(**kwargs)
        if process:
            state = self.replace_2d(state)
            if move is not None:
                move = self.move_index(move)
        return state, turn, move, winner

