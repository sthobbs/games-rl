from agents.Agent_Connect4 import Agent_Connect4
import sys


class Agent_Connect4_User(Agent_Connect4):
    """An agent that allows a human to manually play Connect 4."""

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
        valid_cols = [col+1 for _, col in valid]
        while move not in valid_cols:
            move = input("Enter Valid Move 1-7: ")
            if move in ['q', 'quit', 'exit']:
                print('Exiting...')
                sys.exit(0)
            try:
                move = int(move)
            except ValueError:
                print("Invalid Move, Try Again")
        move -= 1
        move = [(i, j) for i, j in valid if j == move][0]  # convert col to (row, col)
        # play move
        state = self.play_move(state, move)  # play move
        return state, move
