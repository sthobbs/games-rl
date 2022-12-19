from agents.Agent_Connect4 import Agent_Connect4
from mcts.MCTS_Node import MCTS_Node
from mcts.MCTS_Connect4_methods import MCTS_Connect4_methods


class Agent_Connect4_MCTS(Agent_Connect4):
    """Agents plays tic-tac-toe moves based on Monte Carlo Tree Search"""
    def __init__(self, agent_idx=None, simulations=1000, verbose=False):
        super().__init__(agent_idx)
        self.simulations = simulations # number of simulations for MCTS
        self.verbose = verbose # set to True for debugging

    def play_turn(self, state):
        """the Agent plays a turn, and returns the new game state, along with the move played"""
        mcts = Connect4_MCTS_Node(state, turn=self.agent_idx)
        mcts.simulations(self.simulations) # play simulations
        move = mcts.best_move(verbose=self.verbose) # find best most
        state, move = self.play_move(state, move) # play move
        return state, move


class Connect4_MCTS_Node(MCTS_Connect4_methods, MCTS_Node):
    """Monte Carlo Tree Search Node for a game of tic tac toe"""

    def play_move(self, move):
        """play a specific move, returning the new game state (and the move played)."""
        state, move = Agent_Connect4_MCTS(self.turn).play_move(self.state, move, deepcopy_state=True)
        return state, move


