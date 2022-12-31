import random
from agents.Agent_TicTacToe import Agent_TicTacToe


class Agent_TicTacToe_Random(Agent_TicTacToe):
    """Agent that plays tic-tac-toe randomly."""

    def __init__(self, agent_idx=None, **kwargs):
        """
        Initialize the Agent with an agent index.

        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how to mark the game state).
        """
        super().__init__(agent_idx, **kwargs)

    def play_turn(self, state):
        """
        the Agent plays a turn, randomly, and returns the new game state.

        Parameters
        ----------
        state : list of list of str
            the current game state
        """
        valid = self.valid_moves(state)  # get valid moves
        move = random.choice(valid)  # get random valid move
        state = self.play_move(state, move)  # play move
        return state, move
