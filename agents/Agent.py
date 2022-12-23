import abc
import pickle
from tqdm import tqdm
from games.TicTacToe import TicTacToe


class Agent():
    """Abstract class for Agents to play games."""
    
    def __init__(self, agent_idx):
        """
        Initialize the Agent with an agent index.
        
        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how they mark the game state).
        """
        self.agent_idx = agent_idx # the agent index (which often specifies how they mark the game state)
        self.game = TicTacToe

    @abc.abstractmethod
    def play_turn(self, state):
        """
        the Agent plays a turn, and returns the new game state, along with the move played
        
        Parameters
        ----------
        state : any
            the current game state
            
        Returns
        -------
        state : any
            the new game state
        move : any
            the move played
        """

    def format_X_datapoint(self, state, turn):
        """
        Format a datapoint for when generating training data.

        Parameters
        ----------
        state : any
            the current game state
        turn : int
            the current turn
        """
        return state + [turn]

    def to_pickle(self, path):
        """
        Save the agent to a pickle file.
        
        Parameters
        ----------
        path : str
            the path to save the pickle file to
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def from_pickle(self, path):
        """
        Load the agent from a pickle file.
        
        Parameters
        ----------
        path : str
            the path to load the pickle file from
        """
        with open(path) as f:
            self = pickle.load(f)

    # TODO?: parallelize this
    # TODO?: might move this out of the agent class
    def gen_data(self, n, agents, player=None, return_results=False,
                 datapoints_per_game=1, verbose=True, one_hot=True):
        """
        Generate training data by having agents play against each other.

        Parameters
        ----------
        n : int
            number of games to play
        agents : list
            list of agents to play the game
        player : int
            index of agent which we are generating data for.
            if None, data is picked randomly from all players
        return_results : bool
            if True, return the results of the games
        datapoints_per_game : int
            number of datapoints to generate per game
        verbose : bool
            if True, show a progress bar
        one_hot : bool
            if True, encode the game result as a one-hot vector

        Returns
        -------
        Xv : list
            input data for the value model (which predicts P(win))
        Yv : list
            list of game results
        Xp : list
            input data for the policy model (which predicts the next move)
        Yp : list
            list of moves
        """
        # set agent indexes
        for i, agent in enumerate(agents):
            agent.agent_idx = i
        twl = [0, 0, 0] # tie-win-loss count for agents[0]
        Xv, Yv, Xp, Yp = [], [], [], []
        for _ in tqdm(range(n), disable=(not verbose)):
            # set up game
            g = self.game(agents, store_states=True)
            # play game
            g.play_game(pprint=False)
            # get random position from game, and the game result (i.e. which agent won, or draw)
            for _ in range(datapoints_per_game):
                state, turn, move, winner = g.get_data_point(player=player)
                # add to datasets
                x = self.format_X_datapoint(state, turn)
                # set value network response
                if one_hot:
                    if turn == winner: # win
                        yv = [0, 0, 1]
                    elif winner == -1: # tie
                        yv = [0, 1, 0]
                    else: # loss
                        yv = [1, 0, 0]                
                else:
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
            if self.agent_idx == 0:
                results = f"wins = {twl[1]}, ties = {twl[0]}, losses = {twl[2]}"
            else:
                results = f"wins = {twl[2]}, ties = {twl[0]}, losses = {twl[1]}"
            print(results)
        return Xv, Yv, Xp, Yp
