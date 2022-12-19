import random
from agents.Agent_TicTacToe import Agent_TicTacToe

class Agent_TicTacToe_Random(Agent_TicTacToe):

    def __init__(self, agent_idx=None):
        super().__init__(agent_idx)
    
    def play_turn(self, state):
        """the Agent plays a turn, and returns the new game state, along with the move played"""
        valid = self.valid_moves(state) # get valid moves
        move = random.choice(valid) # get random valid move
        state, move = self.play_move(state, move) # play move
        return state, move
