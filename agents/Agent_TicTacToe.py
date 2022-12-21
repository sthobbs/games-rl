from copy import deepcopy
from agents.Agent import Agent
from games.TicTacToe import TicTacToe


class Agent_TicTacToe(Agent):
    """Agents plays tic-tac-toe"""
    def __init__(self, agent_idx=None):
        super().__init__(agent_idx)
        self.game = TicTacToe
    
    def valid_moves(self, state):
        valid = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == ' ':
                    valid.append((i, j))
        return valid

    def play_move(self, state, move, deepcopy_state=False):
        """the Agent plays a specific move for their turn."""
        if deepcopy_state:
            state = deepcopy(state)
        i, j = move
        state[i][j] = self.agent_idx
        return state, move
