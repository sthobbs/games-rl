from agents.Agent_TicTacToe import Agent_TicTacToe
from mcts.MCTS_Node import MCTS_Node
from games.TicTacToe import TicTacToe


class MCTS_TicTacToe_methods():
    """Tic Tac Toe game specific methods Monte Carlo Tree Search Node"""

    def valid_moves(self):
        """return a list of valid next moves for a given game state"""
        return Agent_TicTacToe().valid_moves(self.state)
    
    def game_result(self):
        """ 
        Detmine the winner of the game:
            Return agent index of winning player if a player has won.
            Return None the game is not over.
            Return -1 if the game is over and there is no winner (i.e. a tie).
        """
        return TicTacToe(agents=[None, None], state=self.state).result()



