from copy import deepcopy
from agents.Agent import Agent
from games.Connect4 import Connect4


class Agent_Connect4(Agent):
    """Agents that plays Connect 4."""

    def __init__(self, agent_idx=None, **kwargs):
        """
        Initialize the Agent with an agent index.

        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how to mark the game state).
        """
        super().__init__(agent_idx, **kwargs)
        self.game = Connect4

    def valid_moves(self, state):
        """
        Return a list of valid next moves for a given game state.

        Parameters
        ----------
        state : list of list of str
            the current game state
        """
        valid = []
        for col in range(7):
            for row in range(5, -1, -1):
                if state[row][col] == ' ':
                    valid.append((row, col))
                    break
        return valid

    def play_move(self, state, move, deepcopy_state=False):
        """
        the Agent plays a specific move for their turn.

        Parameters
        ----------
        state : list of list of str
            the current game state
        move : tuple of int
            the move to play
        deepcopy_state : bool
            whether to deepcopy the state before playing the move
        """
        if deepcopy_state:
            state = deepcopy(state)
        i, j = move
        state[i][j] = self.agent_idx
        return state
