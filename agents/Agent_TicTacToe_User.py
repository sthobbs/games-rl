from agents.Agent_TicTacToe import Agent_TicTacToe
# from games.TicTacToe import TicTacToe


class Agent_TicTacToe_User(Agent_TicTacToe):
    """An agent that prompts a human to manually enter moves"""
    def __init__(self, agent_idx=None):
        super().__init__(agent_idx)

    def play_turn(self, state):
        """the Agent plays a turn, and returns the new game state, along with the move played"""
        valid = self.valid_moves(state) # get valid moves
        move = None
        # print([3*i + j for i, j in valid])
        while move not in {3*i + j + 1 for i, j in valid}:
            # TicTacToe(agents=[None, None], state=state).pretty_print() # print current game state for user
            move = int(input('Enter Valid Move 1-9: \n1 2 3\n4 5 6\n7 8 9\n'))
        i, j = divmod(move-1, 3)
        move = (i, j)
        state, move = self.play_move(state, move) # play move
        return state, move
