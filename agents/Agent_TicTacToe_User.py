from agents.Agent_TicTacToe import Agent_TicTacToe


class Agent_TicTacToe_User(Agent_TicTacToe):
    """An agent that allows a human to manually play tic tac toe."""

    def __init__(self, agent_idx=None):
        """
        Initialize the Agent with an agent index.

        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies to they mark the game state).
        """
        super().__init__(agent_idx)

    def play_turn(self, state):
        """
        the Agent plays a turn, and returns the new game state, along with the move played

        Parameters
        ----------
        state : list of list of str
            the current game state
        """
        # get valid moves
        valid = self.valid_moves(state)
        # have human enter valid move
        move = None
        flat_valid_moves = {3*i + j + 1 for i, j in valid}
        while move not in flat_valid_moves:
            move = int(input('Enter Valid Move 1-9: \n1 2 3\n4 5 6\n7 8 9\n'))
        i, j = divmod(move-1, 3)
        move = (i, j)
        # play move
        state = self.play_move(state, move)
        return state, move
