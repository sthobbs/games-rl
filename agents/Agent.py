import abc
import pickle
import torch
from tqdm import tqdm
from games.TicTacToe import TicTacToe
import logging

class Agent():
    """Abstract class for Agents to play games."""

    def __init__(self, agent_idx, logger=None, log_file=None):
        """
        Initialize the Agent with an agent index.

        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how to mark the game state).
        logger : logging.Logger, optional
            the logger to use. If None, a new logger is created.
        log_file : str, optional
            the file to log to. If None, no file logging is performed.
        """
        self.agent_idx = agent_idx
        self.game = TicTacToe
        # set up logging
        if logger is None:
            self.setup_logger(log_file)
        else:
            self.logger = logger

    def setup_logger(self, log_file=None):
        """
        Set up logging for the Agent.

        Parameters
        ----------
        log_file : str, optional
            the file to log to. If None, no file logging is performed.
        """

        # create logger
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__).getChild(str(id(self)))
        self.logger.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # create and add handlers for console output
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # create and add handlers for file output
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

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
        self.logger.info(f"copied agent to {path}")


    def from_pickle(self, path):
        """
        Load the agent from a pickle file into the current instance.

        Parameters
        ----------
        path : str
            the path to load the pickle file from
        """
        with open(path) as f:
            self.__dict__ = pickle.load(f).__dict__
        self.logger.info(f"loaded agent from {path}")
        

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
            if True, return loss, tie, and win counts
        datapoints_per_game : int
            number of datapoints to generate per game
        verbose : bool
            if True, log win, tie, and loss counts
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
        twl = [0, 0, 0]  # tie-win-loss count for agents[0]
        Xv, Yv, Xp, Yp = [], [], [], []
        for _ in tqdm(range(n), disable=(not verbose)):
            # set up game
            g = self.game(agents, store_states=True)
            # play game
            g.play_game(pprint=False)
            # get random position from game, and result (i.e. which agent won, or draw)
            for _ in range(datapoints_per_game):
                state, turn, move, winner = g.get_data_point(player=player)
                # add to datasets
                x = self.format_X_datapoint(state, turn)
                # set value network response
                if one_hot:
                    if turn == winner:  # win
                        yv = [0, 0, 1]
                    elif winner == -1:  # tie
                        yv = [0, 1, 0]
                    else:  # loss
                        yv = [1, 0, 0]
                else:
                    if turn == winner:  # win
                        yv = 1
                    elif winner == -1:  # tie
                        yv = 0
                    else:  # loss
                        yv = -1
                # if there is a next move (i.e. not at the terminal game state)
                if move:
                    yp = move
                    Xp.append(x)
                    Yp.append(yp)
                Xv.append(x)
                Yv.append(yv)
            # update tie-win-loss count for agents[0]
            twl[winner+1] += 1
        # set number of ties, wins, and losses
        t = twl[0]
        if self.agent_idx == 0:
            l, w = twl[2], twl[1]
        else:
            l, w = twl[1], twl[2]
        # log results
        if verbose:
            results = f"wins = {w}, ties = {t}, losses = {l}"
            self.logger.info(results)
        # return results
        if return_results:
            return [l, t, w]
        # return data
        return Xv, Yv, Xp, Yp
