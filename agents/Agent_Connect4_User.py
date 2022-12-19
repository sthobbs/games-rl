from agents.Agent_Connect4 import Agent_Connect4


class Agent_Connect4_User(Agent_Connect4):
    """An agent that prompts a human to manually enter moves"""
    def __init__(self, agent_idx=None):
        super().__init__(agent_idx)

    def play_turn(self, state):
        """the Agent plays a turn, and returns the new game state, along with the move played"""
        valid = self.valid_moves(state) # get valid moves
        move = None
        valid_cols = [col+1 for _, col in valid]
        while move not in valid_cols:
            move = input("Enter Valid Move 1-7: ")
            try:
                move = int(move)
            except:
                ... # TODO: some sort of quit/exit command
        move -= 1
        move = [(i, j) for i, j in valid if j == move][0] # convert col to (row, col)
        state, move = self.play_move(state, move) # play move
        return state, move
