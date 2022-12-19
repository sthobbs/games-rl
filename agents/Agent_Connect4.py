from copy import deepcopy
from agents.Agent import Agent


class Agent_Connect4(Agent):
    """Agents plays Connect 4"""
    def __init__(self, agent_idx=None):
        super().__init__(agent_idx)
    
    def valid_moves(self, state):
        valid = []
        for col in range(7):
            for row in range(5, -1, -1):
                if state[row][col] == ' ':
                    valid.append((row, col))
                    break
        return valid

    def play_move(self, state, move, deepcopy_state=False):
        """the Agent plays a specific move for their turn."""
        if deepcopy_state:
            state = deepcopy(state)
        i, j = move
        state[i][j] = self.agent_idx
        return state, move
