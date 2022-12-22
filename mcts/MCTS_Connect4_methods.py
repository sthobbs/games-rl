from agents.Agent_Connect4 import Agent_Connect4
from games.Connect4 import Connect4


class MCTS_Connect4_methods():
    """Connect 4 game specific methods for Monte Carlo Tree Search Node."""

    def valid_moves(self):
        """Return a list of valid next moves for a given game state."""
        return Agent_Connect4().valid_moves(self.state)
    
    def game_result(self):
        """
        Determine the winner of the game.

        Return
        ------
        None if the game is not over.
        -1 if the game is over and there is no winner (i.e. a tie).
        agent index (i.e. `turn`) of winning player if a player has won.
        """
        return Connect4(agents=[None, None], state=self.state).result()
