import abc
import pickle
from tqdm import tqdm
from games.TicTacToe import TicTacToe


class Agent():
    
    def __init__(self, agent_idx):
        self.agent_idx = agent_idx # the agent index (which often specifies how they mark the game state)
    
    @abc.abstractmethod
    def play_turn(self, state):
        """the Agent plays a turn, and returns the new game state, along with the move played"""
        ...

    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def from_pickle(self, path):
        with open(path) as f:
            self = pickle.load(f)

    # TODO?: parallelize this
    # TODO?: might move this out of the agent class
    def gen_data(self, n, agents, game=TicTacToe, player=None, return_results=False, pprint=False, 
                 datapoints_per_game=1, verbose=True):
        """Generate training data (with n data points) for two models, with n self-play games
            - One model outputs P(win),
            - The other predicts the next move
        `player` index of agent which we are generating data for.
            if None, data is picked randomly from all players
        """ 
        twl = [0, 0, 0] # tie-win-loss count for agents[0]
        Xv, Yv, Xp, Yp = [], [], [], []
        for _ in tqdm(range(n), disable=(not verbose)):
            # set up game
            g = game(agents, store_states=True)
            # play game
            g.play_game(pprint=pprint)
            # get random position from game, and the game result (i.e. which agent won, or draw)
            for _ in range(datapoints_per_game):
                state, turn, move, winner = g.get_data_point(player=player)
                # add to datasets
                x = state + [turn]
                # set value network response
                if turn == winner: # win
                    yv = 1 # [0, 0, 1]
                elif winner == -1: # tie
                    yv = 0 # [0, 1, 0]
                else: # loss
                    yv = -1 # [1, 0, 0]
                if move: # there is only a next move if terminal game state is not picked as the datapoint
                    yp = move # policy network response
                    Xp.append(x)
                    Yp.append(yp)
                Xv.append(x)
                Yv.append(yv)
            # update tie-win-loss count for agents[0]
            twl[winner+1] += 1
        if return_results:
            return twl
        if verbose:
            results = f"wins = {twl[1]}, ties = {twl[0]}, losses = {twl[2]}"
            print(results)
        return Xv, Yv, Xp, Yp