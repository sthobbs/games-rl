from agents.Agent_TicTacToe import Agent_TicTacToe
from mcts.MCTS_Node import MCTS_Node
from mcts.MCTS_TicTacToe_methods import MCTS_TicTacToe_methods


class Agent_TicTacToe_MCTS(Agent_TicTacToe):
    """Agents plays tic-tac-toe moves based on Monte Carlo Tree Search."""

    def __init__(self, agent_idx=None, simulations=1000, verbose=False):
        """
        Initialize the Agent
        
        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how they mark the game state).
        simulations : int
            the number of simulations to run for MCTS.
        verbose : bool
            whether to print debugging information.
        """
        super().__init__(agent_idx)
        self.simulations = simulations # number of simulations for MCTS
        self.verbose = verbose # set to True for debugging

    def play_turn(self, state):
        """
        The Agent plays a turn, based on MCTS, and returns the new game state,
        along with the move played
        
        Parameters
        ----------
        state : list of list of str
            the current game state
        """
        mcts = TicTacToe_MCTS_Node(state, turn=self.agent_idx)
        # play simulations
        mcts.simulations(self.simulations)
        # find best most
        move = mcts.best_move(verbose=self.verbose)
        # play move
        state, move = self.play_move(state, move) 
        return state, move


class TicTacToe_MCTS_Node(MCTS_TicTacToe_methods, MCTS_Node):
    """Monte Carlo Tree Search Node for a game of tic tac toe."""

    def play_move(self, move):
        """
        play a specific move, returning the new game state (and the move played).
        
        Parameters
        ----------
        move : tuple of int
            the move to play
        """
        state, move = Agent_TicTacToe_MCTS(self.turn).play_move(self.state, move, deepcopy_state=True)
        return state, move
