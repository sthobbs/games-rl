import itertools
from games.Game import Game


class TicTacToe(Game):
    def __init__(self, agents, state=None, *args, **kwargs):
        assert len(agents) == 2, f'There should be 2 agents, but there are {len(agents)}.'
        if state is None:
            self.state = [[' ', ' ', ' '], # ' ' => empty
                          [' ', ' ', ' '], # 0 => agent 0 played there
                          [' ', ' ', ' ']] # 1 => agent 1 played there
        else:
            self.state = state
        super().__init__(self.state, agents, *args, **kwargs)

    def result(self):
        """
        Detmine the winner of the game:
            Return agent index (i.e. `turn`) of winning player if a player has won.
            Return None the game is not over.
            Return -1 if the game is over and there is no winner (i.e. a tie).
        """
        # if a player has won, return the index of the winning agent
        for p in range(len(self.agents)):
            if p == self.state[0][0] == self.state[0][1] == self.state[0][2] or\
               p == self.state[1][0] == self.state[1][1] == self.state[1][2] or\
               p == self.state[2][0] == self.state[2][1] == self.state[2][2] or\
               p == self.state[0][0] == self.state[1][0] == self.state[2][0] or\
               p == self.state[0][1] == self.state[1][1] == self.state[2][1] or\
               p == self.state[0][2] == self.state[1][2] == self.state[2][2] or\
               p == self.state[0][0] == self.state[1][1] == self.state[2][2] or\
               p == self.state[0][2] == self.state[1][1] == self.state[2][0]:
                return p
        # if the game is not over, return None
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == ' ':
                    return None
        # if it's a tie, return -1
        return -1

    def pretty_print(self):
        """print game state in human-readable format."""
        print("")
        print(f"{self.state[0][0]} | {self.state[0][1]} | {self.state[0][2]}")
        print(f"{self.state[1][0]} | {self.state[1][1]} | {self.state[1][2]}")
        print(f"{self.state[2][0]} | {self.state[2][1]} | {self.state[2][2]}")

    @staticmethod
    def move_index(move):
        """get index of move if state was flattened"""
        move_idx = 3 * move[0] + move[1]
        return move_idx    

    def get_data_point(self, process=True, **kwargs):
        """
        grab a random game state, and output the tuple (state, turn, move, winner)
            `state` is the current game state
            `turn` is the index of the agent that plays next
            `move` is the move the next player will make
            `winner` is the index of the agent that won, or -1 if the game was a tie.
            `process` = True => replace " "s with -1s, flatten state to a 1d list, and
                ordinal encode move, so these can be used as input to NN training
        this will be used to generate training data.
        """
        state, turn, move, winner = super().get_data_point(**kwargs)
        if process:
            state = self.replace_2d(state)
            state = self.flatten_state(state)
            if move is not None:
                move = self.move_index(move)
        return state, turn, move, winner
