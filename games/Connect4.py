from games.Game import Game


class Connect4(Game):
    """Connect4 game class."""

    def __init__(self, agents, state=None, *args, **kwargs):
        """
        Initialize the game state.

        Parameters
        ----------
        agents : list of Agent
            List of agents playing the game.
        state : list of list of str (optional)
            Game state. If None, initialize to empty board.
        turn : int (optional)
            Index (starting at 0) of agent whose turn it is
        store_states : bool (optional)
            If True, keep a sequence of all (state, turn, last_move) tuples
                Where turn is the player who just played.
                This can be used to generate training data.
        """
        assert len(agents) == 2, f'There should be 2 agents, but there are {len(agents)}.'
        if state is None:
            self.state = [[' ', ' ', ' ', ' ', ' ', ' ', ' '], # ' ' => empty
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '], # 0 => agent 0 played there
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '], # 1 => agent 1 played there
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                          [' ', ' ', ' ', ' ', ' ', ' ', ' '],
                          [' ', ' ', ' ', ' ', ' ', ' ', ' ']] 
        else:
            self.state = state
        super().__init__(self.state, agents, *args, **kwargs)

    def result(self):
        """
        Determine the winner of the game.

        Return
        ------
        None if the game is not over.
        -1 if the game is over and there is no winner (i.e. a tie).
        agent index (i.e. `turn`) of winning player if a player has won.
        """
        # TODO: could optimize by only checking the last move
        # check for winner 
        for col in range(7):
            # set row to be the index of the top move in column col
            # (assume the top move in one of the columns is part of the connect 4 (if it exists))
            # TODO: probably slightly faster to start from bottom
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
        """Print game state in human-readable format."""
        print("\n  1   2   3   4   5   6   7  ") # column numbers for human readability
        for row in self.state:
            print(f"| {' | '.join(str(v) for v in row)} |")

    @staticmethod
    def move_index(move):
        """
        Get column index of move from tuple.
        
        Parameters
        ----------
        move : tuple
            Move to get column index of.
        """
        return move[1]    

    def get_data_point(self, process=True, **kwargs):
        """
        Get a random game state and the next move. This can be used to generate training data.

        Parameters
        ----------
        process : bool
            If True, replace " "s with -1s, and ordinal encode move, so these can be
            used as input to NN training.
        player : int or None
            Can be set to the index of the agent from whom you want to pick a random move from.
            If None, pick a random state from any player

        Returns
        -------
        state : list
            Game state (e.g. 3x3 array for tic tac toe)
        turn : int
            Index (starting at 0) of agent whose turn it is
        move : tuple
            Move the next player will make, (None if it's the terminal game state)
        winner : int
            Index of the agent that won, or -1 if the game was a tie.
        """
        state, turn, move, winner = super().get_data_point(**kwargs)
        if process:
            state = self.replace_2d(state)
            if move is not None:
                move = self.move_index(move)
        return state, turn, move, winner
