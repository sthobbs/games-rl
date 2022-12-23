import random
from agents.Agent_Connect4 import Agent_Connect4

class Agent_Connect4_Random(Agent_Connect4):
    """Agent that plays Connect 4 randomly."""

    def __init__(self, agent_idx=None):
        """
        Initialize the Agent with an agent index.

        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how they mark the game state).
        """
        super().__init__(agent_idx)
    
    def play_turn(self, state):
        """
        the Agent plays a turn, randomly, and returns the new game state,
        along with the move played
        
        Parameters
        ----------
        state : list of list of str
            the current game state
        """
        # get valid moves
        valid = self.valid_moves(state)
        # pick random valid move
        move = random.choice(valid)
        # play move
        state = self.play_move(state, move)
        return state, move
