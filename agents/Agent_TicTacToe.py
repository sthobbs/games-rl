from copy import deepcopy
from agents.Agent import Agent
from games.TicTacToe import TicTacToe


class Agent_TicTacToe(Agent):
    """Agents that plays tic-tac-toe."""

    def __init__(self, agent_idx=None):
        """
        Initialize the Agent with an agent index.
        
        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how they mark the game state).
        """
        super().__init__(agent_idx)
        self.game = TicTacToe
    
    def valid_moves(self, state):
        """
        Return a list of valid next moves for a given game state.
        
        Parameters
        ----------
        state : list of list of str
            the current game state
        """
        valid = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':
                    valid.append((i, j))
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
