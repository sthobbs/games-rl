from agents.Agent_TicTacToe import Agent_TicTacToe
from mcts.MCTS_NN_Node import MCTS_NN_Node
from mcts.MCTS_TicTacToe_methods import MCTS_TicTacToe_methods
from games.TicTacToe import TicTacToe
from games.Game import Game
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

criterion = nn.CrossEntropyLoss()


class Value(nn.Module):
    """Value network for agent, predicts P(win)"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Policy(nn.Module):
    """Policy network for agent, predicts next move"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent_TicTacToe_MCTS_NN(Agent_TicTacToe):
    """
    Agent that plays tic-tac-toe using Monte Carlo Tree Search
    guided by neural networks.
    """

    def __init__(self, agent_idx=None, simulations=1000, depth=None, c=1.41,
                 tau=0.5, n_random=0, verbose=False, value=None, policy=None,
                 **kwargs):
        """
        Initialize agent with value and policy networks.

        Parameters
        ----------
        agent_idx : int
            Index of agent in game.
        simulations : int
            Number of simulations to run for each move.
        depth : int
            Depth of tree to search.
            If None, search until game is over.
        c : float
            Exploration constant.
        tau : float
            Temperature for softmax.
        n_random : int
            Number of random moves to make at each node.
        verbose : bool
            Print information about agent's moves. Useful for debugging.
        value : torch.nn.Module
            Value network. Used to predict P(win) for each state.
        policy : torch.nn.Module
            Policy network. Used to predict next move for each state.
        """
        super().__init__(agent_idx, **kwargs)
        self.simulations = simulations  # number of simulations for MCTS
        self.depth = depth
        self.c = c  # exploration constant
        self.tau = tau  # temperature for softmax
        self.n_random = n_random  # number of initial random moves at each node
        self.verbose = verbose
        self.learning_rate = 0.001
        self.momentum = 0.9
        # set value network
        if value is None:
            self.value = Value()
            self.value.name = 'value'  # name model for reference later
        else:
            self.value = value
        assert hasattr(self, 'value'), 'Invalid value network'
        # set policy network
        if policy is None:
            self.policy = Policy()
            self.policy.name = 'policy'  # name model for reference later
        else:
            self.policy = policy
        assert hasattr(self, 'policy'), 'Invalid policy network'
        # game data
        self.Xv = torch.tensor([], dtype=torch.float32)
        self.yv = torch.tensor([], dtype=torch.float32)
        self.Xp = torch.tensor([], dtype=torch.float32)
        self.yp = torch.tensor([], dtype=torch.long)

    def deepcopy_without_data(self):
        """
        Return deepcopy of agent without game data.
        """
        agent = Agent_TicTacToe_MCTS_NN()
        agent.__dict__ = {k: v for k, v in self.__dict__.items() if k not in ['Xv', 'yv', 'Xp', 'yp']}
        return deepcopy(agent)

    def iter_minibatches(self, X, y, batch_size=32):
        """
        iterate over minibatches.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        y : torch.Tensor
            Output data.
        batch_size : int
            Size of minibatches.
        """
        # Provide chunks one by one
        cur = 0
        while cur < len(y):
            chunkrows = slice(cur, cur + batch_size)
            X_chunk, y_chunk = X[chunkrows], y[chunkrows]
            yield X_chunk, y_chunk
            cur += batch_size

    def fit_epoch(self, X, y, model, batch_size=32):
        """
        Fit one epoch.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        y : torch.Tensor
            Output data.
        model : torch.nn.Module
            Model to fit.
        batch_size : int
            Size of minibatches.
        """
        # set optimizer
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate,
                              momentum=self.momentum)
        # randomly permute data
        X, y = self.random_permute(X, y)
        # iterate over minibatches
        for X_chunk, y_chunk in self.iter_minibatches(X, y, batch_size=batch_size):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(X_chunk)
            loss = criterion(outputs, y_chunk)
            loss.backward()
            optimizer.step()

    def random_permute(self, X, y):
        """
        Randomly permute data.

        Parameters
        ----------
        X : torch.Tensor
            Input data.
        y : torch.Tensor
            Output data.
        """
        idx = torch.randperm(y.shape[0])
        X = X[idx].view(X.size())
        y = y[idx].view(y.size())
        return X, y

    def fit_model(self, X_train, y_train, X_test, y_test, model, early_stopping=None,
                  verbose=10, num_epochs=10):
        """
        Fit either value or policy network for num_epochs epochs.

        Parameters
        ----------
        X_train : torch.Tensor
            Input data for training.
        y_train : torch.Tensor
            Output data for training.
        X_test : torch.Tensor
            Input data for testing.
        y_test : torch.Tensor
            Output data for testing.
        model : torch.nn.Module
            Model to fit.
        early_stopping : int
            Number of epochs to wait before stopping if no improvement.
            If None, do not use early stopping.
        verbose : int
            Log metrics every verbose epochs.
        num_epochs : int
            Number of epochs to train for.
        """
        assert model.name in ['value', 'policy'], "model name not in ['value', 'policy']"
        if early_stopping is None:
            early_stopping = float("inf")
        # loop over the dataset multiple times
        for epoch in range(num_epochs):
            # fit 1 epoch
            self.fit_epoch(X_train, y_train, model)
            # evaluate model
            y_pred = model(X_train)
            y_true = y_train
            train_loss = criterion(y_pred, y_true).item()
            y_pred = model(X_test)
            y_true = y_test
            test_loss = criterion(y_pred, y_true).item()
            if epoch % verbose == 0:  # log metrics
                self.logger.info(f"[{epoch}] train loss = {train_loss}, test loss = {test_loss}")
            # check best log loss so far
            if epoch == 0:
                min_log_loss = test_loss  # min test loss so far
                min_log_loss_train = train_loss  # associated train loss
                min_log_loss_epoch = 0  # epoch with min test loss
                min_log_loss_model = deepcopy(model)  # model with min test loss
            elif test_loss < min_log_loss:
                min_log_loss = test_loss
                min_log_loss_train = train_loss
                min_log_loss_epoch = epoch
                min_log_loss_model = deepcopy(model)
            # early stopping
            if epoch > min_log_loss_epoch + early_stopping:
                break
        # log optimal epoch
        if early_stopping < float("inf"):
            self.logger.info("optimal epoch:")
            epoch = min_log_loss_epoch
            train_loss, test_loss = min_log_loss_train, min_log_loss
            self.logger.info(f"[{epoch}] train loss = {train_loss}, test loss = {test_loss}")
            # copy optimal model to agent
            if model.name == 'value':
                self.value = min_log_loss_model
            elif model.name == 'policy':
                self.policy = min_log_loss_model
        self.logger.info("")

    def gen_data_diff_ops(self, n, ops, datapoints_per_game=1, n_jobs=1):
        """
        play n games against different opponents, randomly selected from ops,
        and generate data from these games.

        Parameters
        ----------
        n : int
            Number of games to play.
        ops : list
            List of opponent agents.
        datapoints_per_game : int
            Number of datapoints to generate per game.
        n_jobs : int
            Number of games to play in parallel.
        """
        n_datapoints = n * datapoints_per_game * 8  # 8 augmentations
        self.logger.info(f'generating {n_datapoints} datapoint from {n} games')
        Xv, yv, Xp, yp = [], [], [], []

        # parallelize games
        if n_jobs > 1:
            # copy agent without data before passing to pool
            agent_copy = self.deepcopy_without_data()
            # set up kwargs generator for parallel games
            kwargs_gen = agent_copy.kwargs_generator_for_gen_data(n, ops, datapoints_per_game)
            # play games in parallel
            with Pool(n_jobs) as pool:
                game_data = tqdm(pool.imap(agent_copy.gen_data_kwargs, kwargs_gen), total=n)
                # collect data
                for Xv_, yv_, Xp_, yp_ in game_data:
                    Xv.extend(Xv_)
                    yv.extend(yv_)
                    Xp.extend(Xp_)
                    yp.extend(yp_)
        
        # play games sequentially
        else:
            for _ in tqdm(range(n)):
                # pick random opponent
                op = random.choice(ops)
                # randomly pick who goes first
                if random.random() < 0.5:
                    agents = [self, op]
                    player = 0
                else:
                    agents = [op, self]
                    player = 1
                # generate data
                Xv_, yv_, Xp_, yp_ = self.gen_data(1, agents=agents, player=player,
                                                datapoints_per_game=datapoints_per_game,
                                                verbose=False)
                Xv.extend(Xv_)
                yv.extend(yv_)
                Xp.extend(Xp_)
                yp.extend(yp_)
        # augment data (with rotations and reflections)
        Xv, yv, Xp, yp = self.augment_data(Xv, yv, Xp, yp)
        # convert to tensors
        Xv = torch.tensor(Xv, dtype=torch.float32)
        yv = torch.tensor(yv, dtype=torch.float32)
        Xp = torch.tensor(Xp, dtype=torch.float32)
        yp = torch.tensor(yp, dtype=torch.long)
        # append to game data
        self.Xv = torch.cat([self.Xv, Xv])
        self.yv = torch.cat([self.yv, yv])
        self.Xp = torch.cat([self.Xp, Xp])
        self.yp = torch.cat([self.yp, yp])

    def augment_data(self, Xv, yv, Xp, yp):
        """
        Augment data. Since tic tac toe is invariant to rotations and reflections,
        return all rotations and reflections of data.

        Parameters
        ----------
        Xv : list
            List of value network input data.
        yv : list
            List of value network output data.
        Xp : list
            List of policy network input data.
        yp : list
            List of policy network output data.
        """
        # Augment Value network
        new_Xv, new_yv = Xv[:], yv[:]
        for X, y in zip(Xv, yv):
            # reflect
            X_, _ = self.reflect(X)
            new_Xv.append(X_)
            new_yv.append(y)
            # rotate
            X, _ = self.rotate(X)
            new_Xv.append(X)
            new_yv.append(y)
            # rotate + reflect
            X_, _ = self.reflect(X)
            new_Xv.append(X_)
            new_yv.append(y)
            # rotate x2
            X, _ = self.rotate(X)
            new_Xv.append(X)
            new_yv.append(y)
            # rotate x2 + reflect
            X_, _ = self.reflect(X)
            new_Xv.append(X_)
            new_yv.append(y)
            # rotate x3
            X, _ = self.rotate(X)
            new_Xv.append(X)
            new_yv.append(y)
            # rotate x3 + reflect
            X_, _ = self.reflect(X)
            new_Xv.append(X_)
            new_yv.append(y)
        # Augment Policy network
        new_Xp, new_yp = Xp[:], yp[:]
        for X, y in zip(Xp, yp):
            # reflect
            X_, y_ = self.reflect(X, y)
            new_Xp.append(X_)
            new_yp.append(y_)
            # rotate
            X, y = self.rotate(X, y)
            new_Xp.append(X)
            new_yp.append(y)
            # rotate + reflect
            X_, y_ = self.reflect(X, y)
            new_Xp.append(X_)
            new_yp.append(y_)
            # rotate x2
            X, y = self.rotate(X, y)
            new_Xp.append(X)
            new_yp.append(y)
            # rotate x2 + reflect
            X_, y_ = self.reflect(X, y)
            new_Xp.append(X_)
            new_yp.append(y_)
            # rotate x3
            X, y = self.rotate(X, y)
            new_Xp.append(X)
            new_yp.append(y)
            # rotate x3 + reflect
            X_, y_ = self.reflect(X, y)
            new_Xp.append(X_)
            new_yp.append(y_)
        return new_Xv, new_yv, new_Xp, new_yp

    def rotate(self, X, y=0):
        """
        Rotate Xv (or Xp) and yp 90 degrees to the right.

        Parameters
        ----------
        X : list
            List of input data.
        y : int
            Policy network output.
        """
        X_ = [X[6], X[3], X[0], X[7], X[4], X[1], X[8], X[5], X[2], X[9]]
        y_ = [2, 5, 8, 1, 4, 7, 0, 3, 6][y]
        return X_, y_

    def reflect(self, X, y=0):
        """
        Reflect Xv (or Xp) and yp along the veritcal line of symmetry.

        Parameters
        ----------
        X : list
            List of input data.
        y : int
            Policy network output.
        """
        X_ = [X[2], X[1], X[0], X[5], X[4], X[3], X[8], X[7], X[6], X[9]]
        y_ = [2, 1, 0, 5, 4, 3, 8, 7, 6][y]
        return X_, y_

    def fit(self, refit_datapoints=200000, test_size=0.3, **kwargs):
        """
        Fit the value and policy networks.

        Parameters
        ----------
        refit_datapoints : int
            Number of data points to use for refitting.
        test_size : float
            Fraction of data to use for testing.
        """
        assert 0 < test_size < 1, 'invalid test size'
        # get most recent data
        Xv = self.Xv[-refit_datapoints:]
        yv = self.yv[-refit_datapoints:]
        Xp = self.Xp[-refit_datapoints:]
        yp = self.yp[-refit_datapoints:]
        # permute data
        Xv, yv = self.random_permute(Xv, yv)
        Xp, yp = self.random_permute(Xp, yp)
        # split data into training and testing sets
        split = int(test_size * yv.shape[0])
        Xv_train, Xv_test = Xv[:split], Xv[split:]
        yv_train, yv_test = yv[:split], yv[split:]
        split = int(test_size * yp.shape[0])
        Xp_train, Xp_test = Xp[:split], Xp[split:]
        yp_train, yp_test = yp[:split], yp[split:]
        # fit value network
        self.logger.info('fitting value network')
        self.fit_model(Xv_train, yv_train, Xv_test, yv_test, model=self.value, **kwargs)
        # fit policy network
        self.logger.info('fitting policy network')
        self.fit_model(Xp_train, yp_train, Xp_test, yp_test, model=self.policy, **kwargs)

    def play_turn(self, state):
        """
        The Agent plays a turn, and returns the new game state,
        along with the move played.

        Parameters
        ----------
        state : list
            The current game state.
        """
        mcts = TicTacToe_MCTS_NN_Node(agent=self, state=state, turn=self.agent_idx,
                                      depth=self.depth, c=self.c, tau=self.tau,
                                      n_random=self.n_random)
        mcts.simulations(self.simulations)  # play simulations
        move = mcts.best_move(verbose=self.verbose)  # find best most
        state = self.play_move(state, move)  # play move
        return state, move


class TicTacToe_MCTS_NN_Node(MCTS_TicTacToe_methods, MCTS_NN_Node):
    """Neural Network Monte Carlo Tree Search Node for a game of tic tac toe."""

    def __init__(self, agent, *args, **kwargs):
        """
        Initialize the node.

        Parameters
        ----------
        agent : Agent_TicTacToe_MCTS_NN
            The agent that is playing the game.
        """
        super().__init__(agent=agent, *args, **kwargs)
        self.agent = agent

    def play_move(self, move):
        """
        Play a specific move, returning the new game state.

        Parameters
        ----------
        move : tuple of int
            The move to play.
        """
        state = Agent_TicTacToe_MCTS_NN(
            agent_idx=self.turn,
            value=self.agent.value,
            policy=self.agent.policy
            ).play_move(self.state, move, deepcopy_state=True)
        return state

    def value_predict(self):
        """Return output of the value network for the current state."""
        x = self.prep_data(self.state, self.turn)
        return F.softmax(self.agent.value(x), dim=1)[0]

    def policy_predict(self):
        """
        Return output of policy networks for the parent's state,
        parent's turn, and last move played.
        """
        move_idx = TicTacToe.move_index(self.last_move)
        x = self.prep_data(self.parent.state, self.parent.turn)
        return F.softmax(self.agent.policy(x), dim=1)[0][move_idx]

    def prep_data(self, state, turn):
        """
        Prepare input for policy or value network (it's the same input)

        Parameters
        ----------
        state : list
            The current game state.
        turn : int
            The current player's turn.
        """
        state = Game.replace_2d(state)
        state = Game.flatten_state(state)
        x = state + [turn]
        x = torch.tensor([x], dtype=torch.float32)
        return x
