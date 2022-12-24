from games.Game import Game


class TicTacToe(Game):
    """TicTacToe game class."""

    def __init__(self, agents, state=None, *args, **kwargs):
        """
        Initialize the game state.

        Parameters
        ----------
        agents : list of Agent
            List of agents playing the game.
        state : list of list of str
            Game state. If None, initialize to empty board.
        """
        assert len(agents) == 2, f'There should be 2 agents, but there are {len(agents)}.'
        if state is None:
            self.state = [[' ', ' ', ' '],  # ' ' => empty
                          [' ', ' ', ' '],  # 0 => agent 0 played there
                          [' ', ' ', ' ']]  # 1 => agent 1 played there
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
        # only check the last move if this information is available
        if self.last_move is not None:
            x, y = self.last_move
            val = self.state[x][y]
            # check if the last move won the game
            if val == self.state[x][0] == self.state[x][1] == self.state[x][2] or\
               val == self.state[0][y] == self.state[1][y] == self.state[2][y] or\
               val == self.state[0][0] == self.state[1][1] == self.state[2][2] or\
               val == self.state[0][2] == self.state[1][1] == self.state[2][0]:
                return val
        # otherwise, check for all possible moves
        else:
            # if a player has won, return the index of the winning agent
            for p in range(len(self.agents)):
                if p == self.state[0][0] == self.state[0][1] == self.state[0][2] or\
                p == self.state[1][0] == self.state[1][1] == self.state[1][2] or\
                p == self.state[2][0] == self.state[2][1] == self.state[2][2] or\
                p == self.state[0][0] == self.state[1][0] == self.state[2][0] or\
                p == self.state[0][1] == self.state[1][1] == self.state[2][1] or\
                p == self.state[0][2] == self.state[1][2] == self.state[2][2] or\
                p == self.state[0][0] == self.state[1][1] == self.state[2][2] or\
                p == self.state[0][2] == self.state[1][1] == self.state[2][0]:
                    return p
        # if the game is not over, return None
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == ' ':
                    return None
        # if it's a tie, return -1
        return -1

    def pretty_print(self):
        """Print game state in human-readable format."""
        print("")
        for row in self.state:
            print(f"{row[0]} | {row[1]} | {row[2]}")

    @staticmethod
    def move_index(move):
        """
        Get index of move if state was flattened

        Parameters
        ----------
        move : tuple of int
            Move to get index of.
        """
        move_idx = 3 * move[0] + move[1]
        return move_idx

    def get_data_point(self, process=True, **kwargs):
        """
        Construct data point from a random saved state. This can be used
        to generate training data.

        Parameters
        ----------
        process : bool
            If True, replace " "s with -1s, flatten state to a 1d list, and
            ordinal encode move, so these can be used as input to NN training
        player : int or None
            Can be set to the index of the agent to pick a random move from.
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
            state = self.flatten_state(state)
            if move is not None:
                move = self.move_index(move)
        return state, turn, move, winner
