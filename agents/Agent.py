from tqdm import tqdm
from games.TicTacToe import TicTacToe
from multiprocessing import Pool
import abc
import pickle
import logging
import random


class Agent():
    """Abstract class for Agents to play games."""

    def __init__(self, agent_idx=0, logger=None, **kwargs):
        """
        Initialize the Agent with an agent index.

        Parameters
        ----------
        agent_idx : int
            the agent index (which often specifies how to mark the game state).
        logger : logging.Logger, optional
            the logger to use. If None, a new logger is created.
        """
        self.agent_idx = agent_idx
        self.game = TicTacToe
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

    def _gen_data_game(self, args):
        """
        Generate training data for one game by having agents play against each other.
        
        Parameters
        ----------
        args : tuple
            tuple of arguments to pass to the function, in the following order:
                agents (args[0]) : list
                    list of agents to play the game
                player (args[1]) : int
                    index of agent which we are generating data for.
                    if None, data is picked randomly from all players
                datapoints_per_game (args[2]) : int
                    number of datapoints to generate per game
                one_hot (args[3]) : bool
                    if True, encode the game result as a one-hot vector
        """
        agents, player, datapoints_per_game, one_hot = args
        Xv, Yv, Xp, Yp = [], [], [], []
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
        return Xv, Yv, Xp, Yp, winner

    def gen_data(self, n, agents, player=None, return_results=False,
                 datapoints_per_game=1, verbose=True, one_hot=True,
                 n_jobs=1):
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
        n_jobs : int
            number of processes to use for parallelization

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

        # parallelize games
        if n_jobs > 1:
            
            # if agent stores data, then make a copy without data before passing to pool
            if hasattr(self, "deepcopy_without_data"):
                agent_copy = self.deepcopy_without_data()
            else:
                agent_copy = self
            agents_copy = []
            for agent in agents:
                if hasattr(agent, "deepcopy_without_data"):
                    agents_copy.append(agent.deepcopy_without_data())
                else:
                    agents_copy.append(agent)

            # set up args for parallel games
            args = (agents_copy, player, datapoints_per_game, one_hot)
            arg_list = [args] * n
            
            # run games in parallel
            with Pool(n_jobs) as pool:
                game_data = tqdm(pool.imap(agent_copy._gen_data_game, arg_list), total=n)
                # collect data
                for Xv_, Yv_, Xp_, Yp_, winner in game_data:
                    Xv.extend(Xv_)
                    Yv.extend(Yv_)
                    Xp.extend(Xp_)
                    Yp.extend(Yp_)
                    # update tie-win-loss count for agents[0]
                    twl[winner+1] += 1
        # run games sequentially
        else:
            for _ in tqdm(range(n), disable=(not verbose)):
                args = (agents, player, datapoints_per_game, one_hot)
                Xv_, Yv_, Xp_, Yp_, winner = self._gen_data_game(args)
                Xv.extend(Xv_)
                Yv.extend(Yv_)
                Xp.extend(Xp_)
                Yp.extend(Yp_)
                # update tie-win-loss count for agents[0]
                twl[winner+1] += 1
        # set number of ties, wins, and losses for current agent
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

    def gen_data_kwargs(self, kwargs):
        """
        Wrapper for gen_data() which allows for passing kwargs as a dict.
        """
        return self.gen_data(**kwargs)


    def kwargs_generator_for_gen_data(self, n, ops, datapoints_per_game):
        """
        generate kwargs for gen_data_kwargs for n games against different
        opponents, randomly selected from ops.
        
        Parameters
        ----------
        n : int
            Number of games to play.
        ops : list
            List of opponent agents.
        datapoints_per_game : int
            Number of datapoints to generate per game.
        """
        for _ in range(n):
            # pick random opponent
            op = random.choice(ops)
            # randomly pick who goes first
            if random.random() < 0.5:
                agents = [self, op]
                player = 0
            else:
                agents = [op, self]
                player = 1
            # yield kwargs for gen_data_kwargs
            yield {
                'n': 1,
                'agents': agents,
                'player': player,
                'datapoints_per_game': datapoints_per_game,
                'verbose': False,
            }
