from agents.Agent_Connect4 import Agent_Connect4
from games.Connect4 import Connect4


class MCTS_Connect4_methods():
    """Connect 4 game specific methods for Monte Carlo Tree Search Node"""

    def valid_moves(self):
        """return a list of valid next moves for a given game state"""
        return Agent_Connect4().valid_moves(self.state)
    
    def game_result(self):
        """ 
        Detmine the winner of the game:
            Return agent index of winning player if a player has won.
            Return None the game is not over.
            Return -1 if the game is over and there is no winner (i.e. a tie).
        """
        return Connect4(agents=[None, None], state=self.state).result()



