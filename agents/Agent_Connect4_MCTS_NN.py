from agents.Agent_Connect4 import Agent_Connect4
from mcts.MCTS_NN_Node import MCTS_NN_Node
from mcts.MCTS_Connect4_methods import MCTS_Connect4_methods
from games.Connect4 import Connect4
from games.Game import Game
from copy import deepcopy
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

criterion = nn.CrossEntropyLoss()


# TODO? could try adding fc layer of (flattened) raw state values
class Value(nn.Module):
    """Value network for agent, predicts P(win)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 120, 4)  # 6x7 -> 3x4x120 ((4x4+1) x 120 parameters)
        self.conv2 = nn.Conv2d(120, 5, 2)  # 3x4x120 -> 2x3x5 ((2x2x120+1) x 5 parameters)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(31, 20)
        self.fc2 = nn.Linear(20, 3)  # output loss, tie, win probabilities

    def forward(self, state, turn):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.cat((x, turn), dim=1)  # concatenate conv output and turn
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Policy(nn.Module):
    """Policy network for agent, predicts next move"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 120, 4)  # 6x7 -> 3x4x120 ((4x4+1) x 120 parameters)
        self.conv2 = nn.Conv2d(120, 5, 2)  # 3x4x120 -> 2x3x5 ((2x2x120+1) x 5 parameters)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(31, 20)
        self.fc2 = nn.Linear(20, 7)  # output (column) move predictions

    def forward(self, state, turn):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.cat((x, turn), dim=1)  # concatenate conv output and turn
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Agent_Connect4_MCTS_NN(Agent_Connect4):
    """
    Agent that plays Connect 4 using Monte Carlo Tree Search
    guided by neural networks.
    """

    def __init__(self, agent_idx=None, simulations=1000, depth=None, c=1.41,
                 tau=0.5, n_random=0, verbose=False, value=None, policy=None):
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
        super().__init__(agent_idx)
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
        self.Xv1 = torch.tensor([], dtype=torch.float32)
        self.Xv2 = torch.tensor([], dtype=torch.float32)
        self.yv = torch.tensor([], dtype=torch.float32)
        self.Xp1 = torch.tensor([], dtype=torch.float32)
        self.Xp2 = torch.tensor([], dtype=torch.float32)
        self.yp = torch.tensor([], dtype=torch.long)

    def deepcopy_without_data(self):
        """
        Return deepcopy of agent without game data.
        """
        agent = Agent_Connect4_MCTS_NN()
        agent.__dict__ = {k: v for k, v in self.__dict__.items() \
            if k not in ['Xv1', 'Xv2' 'yv', 'Xp1', 'Xp2', 'yp']}
        return deepcopy(agent)

    def iter_minibatches(self, X1, X2, y, batch_size=32):
        """
        iterate over minibatches.

        Parameters
        ----------
        X1 : torch.Tensor
            Game state input data.
        X2 : torch.Tensor
            Agent turn input data.
        y : torch.Tensor
            Output data.
        batch_size : int
            Size of minibatches.
        """
        # yield chunks one by one
        cur = 0
        while cur < len(y):
            chunkrows = slice(cur, cur + batch_size)
            X1_chunk, X2_chunk, y_chunk = X1[chunkrows], X2[chunkrows], y[chunkrows]
            X1_chunk = torch.reshape(X1_chunk, (-1, 1, 6, 7))
            yield X1_chunk, X2_chunk, y_chunk
            cur += batch_size

    def fit_epoch(self, X1, X2, y, model, batch_size=32):
        """
        Fit one epoch.

        Parameters
        ----------
        X1 : torch.Tensor
            Game state input data.
        X2 : torch.Tensor
            Agent turn input data.
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
        X1, X2, y = self.random_permute(X1, X2, y)
        # iterate over minibatches
        for X1_, X2_, y_ in self.iter_minibatches(X1, X2, y, batch_size=batch_size):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(X1_, X2_)  # extra bracket for 1-dim input channel
            # breakpoint()
            loss = criterion(outputs, y_)
            loss.backward()
            optimizer.step()

    # TODO: move to agent or utils (need to generalize to list of tensors)
    def random_permute(self, X1, X2, y):
        """
        Randomly permute data.

        Parameters
        ----------
        X1 : torch.Tensor
            Game state input data.
        X2 : torch.Tensor
            Agent turn input data.
        y : torch.Tensor
            Output data.
        """
        idx = torch.randperm(y.shape[0])
        X1 = X1[idx].view(X1.size())
        X2 = X2[idx].view(X2.size())
        y = y[idx].view(y.size())
        return X1, X2, y

    def fit_model(self, X1_train, X2_train, y_train, X1_test, X2_test, y_test,
                  model, early_stopping=None, verbose=10, num_epochs=10):
        """
        Fit either value or policy network for num_epochs epochs.

        Parameters
        ----------
        X1_train : torch.Tensor
            Game state input data for training.
        X2_train : torch.Tensor
            Agent turn input data for training.
        y_train : torch.Tensor
            Output data for training.
        X1_test : torch.Tensor
            Game state input data for testing.
        X2_test : torch.Tensor
            Agent turn input data for testing.
        y_test : torch.Tensor
            Output data for testing.
        model : torch.nn.Module
            Model to fit.
        early_stopping : int
            Number of epochs to wait before stopping if no improvement.
            If None, do not use early stopping.
        verbose : int
            Print metrics every verbose epochs.
        num_epochs : int
            Number of epochs to fit.
        """
        assert model.name in ['value', 'policy'], "model name not in ['value', 'policy']"
        if early_stopping is None:
            early_stopping = float("inf")
        # loop over the dataset multiple times
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            # fit 1 epoch
            self.fit_epoch(X1_train, X2_train, y_train, model)
            # print metrics
            X1_train = torch.reshape(X1_train, (-1, 1, 6, 7))
            y_pred = model(X1_train, X2_train)
            y_true = y_train
            train_loss = criterion(y_pred, y_true).item()
            X1_test = torch.reshape(X1_test, (-1, 1, 6, 7))
            y_pred = model(X1_test, X2_test)
            y_true = y_test
            test_loss = criterion(y_pred, y_true).item()
            if epoch % verbose == 0:  # print metrics
                print(f"[{epoch}] train loss = {train_loss}, test loss = {test_loss}")
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
        # print optimal epoch
        if early_stopping < float("inf"):
            print("optimal epoch:")
            epoch = min_log_loss_epoch
            train_loss, test_loss = min_log_loss_train, min_log_loss
            print(f"[{epoch}] train loss = {train_loss}, test loss = {test_loss}")
            # copy optimal model to agent
            if model.name == 'value':
                self.value = min_log_loss_model
            elif model.name == 'policy':
                self.policy = min_log_loss_model
        print("")

    def gen_data_diff_ops(self, n, ops, datapoints_per_game=1):
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

        Returns
        -------
        Xv1 : torch.Tensor
            Game state value network input data.
        Xv2 : torch.Tensor
            Agent turn value network input data.
        yv : torch.Tensor
            Value network output data.
        Xp1 : torch.Tensor
            Game state policy network input data.
        Xp2 : torch.Tensor
            Agent turn policy network input data.
        yp : torch.Tensor
            Policy network output data.
        """
        Xv, yv, Xp, yp = [], [], [], []
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
            Xv_, yv_, Xp_, yp_ = self.gen_data(1, agents=agents, player=player,
                                               datapoints_per_game=datapoints_per_game,
                                               verbose=False)
            Xv.extend(Xv_)
            yv.extend(yv_)
            Xp.extend(Xp_)
            yp.extend(yp_)
        # reformat data (since we need multiple input tensors)
        Xv1 = [state for state, _ in Xv]
        Xv2 = [turn for _, turn in Xv]
        Xp1 = [state for state, _ in Xp]
        Xp2 = [turn for _, turn in Xp]
        # augment data (since Connect 4 is invariant to vertical reflections)
        Xv1, Xv2, yv, Xp1, Xp2, yp = self.augment_data(Xv1, Xv2, yv, Xp1, Xp2, yp)
        # convert to tensors
        Xv1 = torch.tensor(Xv1, dtype=torch.float32)
        Xv2 = torch.tensor(Xv2, dtype=torch.float32)
        yv = torch.tensor(yv, dtype=torch.float32)
        Xp1 = torch.tensor(Xp1, dtype=torch.float32)
        Xp2 = torch.tensor(Xp2, dtype=torch.float32)
        yp = torch.tensor(yp, dtype=torch.long)
        # append to game data
        self.Xv1 = torch.cat([self.Xv1, Xv1])
        self.Xv2 = torch.cat([self.Xv2, Xv2])
        self.yv = torch.cat([self.yv, yv])
        self.Xp1 = torch.cat([self.Xp1, Xp1])
        self.Xp2 = torch.cat([self.Xp2, Xp2])
        self.yp = torch.cat([self.yp, yp])

    def augment_data(self, Xv1, Xv2, yv, Xp1, Xp2, yp):
        """
        Since Connect 4 is invariant to vertical reflections,
        add this reflection to the dataset.

        Parameters
        ----------
        Xv1 : list
            Game state value network input data.
        Xv2 : list
            Agent turn value network input data.
        yv : list
            Value network output data.
        Xp1 : list
            Game state policy network input data.
        Xp2 : list
            Agent turn policy network input data.
        yp : list
            Policy network output data.
        """
        # Augment Value network
        new_Xv1, new_Xv2, new_yv = Xv1[:], Xv2[:], yv[:]
        for X1, X2, y in zip(Xv1, Xv2, yv):
            X1_, _ = self.reflect(X1)
            new_Xv1.append(X1_)
            new_Xv2.append(X2)
            new_yv.append(y)
        # Augment Policy network
        new_Xp1, new_Xp2, new_yp = Xp1[:], Xp2[:], yp[:]
        for X1, X2, y in zip(Xp1, Xp2, yp):
            X1_, y_ = self.reflect(X1, y)
            new_Xp1.append(X1_)
            new_Xp2.append(X2)
            new_yp.append(y_)
        return new_Xv1, new_Xv2, new_yv, new_Xp1, new_Xp2, new_yp

    def reflect(self, X, y=0):
        """
        Reflect Xv (or Xp) and yp along the veritcal line of symmetry.

        Parameters
        ----------
        X : list
            Game state value network input data.
        y : int
            Policy network output data.
        """
        X_ = []
        for row in X:
            X_.append(row[::-1])
        y_ = 6 - y
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
        # generate training and testing data from different games to avoid leakage
        assert 0 < test_size < 1, 'invalid test size'
        # get most recent data
        Xv1 = self.Xv1[-refit_datapoints:]
        Xv2 = self.Xv2[-refit_datapoints:]
        yv = self.yv[-refit_datapoints:]
        Xp1 = self.Xp1[-refit_datapoints:]
        Xp2 = self.Xp2[-refit_datapoints:]
        yp = self.yp[-refit_datapoints:]
        # permute data
        Xv1, Xv2, yv = self.random_permute(Xv1, Xv2, yv)
        Xp1, Xp2, yp = self.random_permute(Xp1, Xp2, yp)
        # split data into training and testing sets
        split = int(test_size * yv.shape[0])
        Xv1_train, Xv1_test = Xv1[:split], Xv1[split:]
        Xv2_train, Xv2_test = Xv2[:split], Xv2[split:]
        yv_train, yv_test = yv[:split], yv[split:]
        split = int(test_size * yp.shape[0])
        Xp1_train, Xp1_test = Xp1[:split], Xp1[split:]
        Xp2_train, Xp2_test = Xp2[:split], Xp2[split:]
        yp_train, yp_test = yp[:split], yp[split:]
        # fit value network
        print('fitting value network')
        self.fit_model(Xv1_train, Xv2_train, yv_train, Xv1_test, Xv2_test, yv_test,
                       model=self.value, **kwargs)
        # fit policy network
        print('fitting policy network')
        self.fit_model(Xp1_train, Xp2_train, yp_train, Xp1_test, Xp2_test, yp_test,
                       model=self.policy, **kwargs)

    def play_turn(self, state):
        """
        The Agent plays a turn, and returns the new game state,
        along with the move played.

        Parameters
        ----------
        state : list
            The current game state.
        """
        mcts = Connect4_MCTS_NN_Node(agent=self, state=state, turn=self.agent_idx,
                                     depth=self.depth, c=self.c, tau=self.tau,
                                     n_random=self.n_random)
        mcts.simulations(self.simulations)  # play simulations
        move = mcts.best_move(verbose=self.verbose)  # find best most
        state = self.play_move(state, move)  # play move
        return state, move

    def format_X_datapoint(self, state, turn):
        """
        Format a datapoint for when generating training data.

        Parameters
        ----------
        state : list
            the current game state
        turn : int
            the current turn
        """
        return [state, [turn]]


class Connect4_MCTS_NN_Node(MCTS_Connect4_methods, MCTS_NN_Node):
    """Neural Network Monte Carlo Tree Search Node for a game of Connect 4."""

    def __init__(self, agent, *args, **kwargs):
        """
        Initialize the node.

        Parameters
        ----------
        agent : Agent_Connect4_MCTS_NN
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
        state = Agent_Connect4_MCTS_NN(
            agent_idx=self.turn,
            value=self.agent.value,
            policy=self.agent.policy
            ).play_move(self.state, move, deepcopy_state=True)
        return state

    def value_predict(self):
        """Return output of the value network for the current state."""
        x1, x2 = self.prep_data(self.state, self.turn)
        # breakpoint()
        return F.softmax(self.agent.value(x1, x2), dim=1)[0]

    def policy_predict(self):
        """
        Return output of policy networks for the parent's state,
        parent's turn, and last move played.
        """
        move_idx = Connect4.move_index(self.last_move)
        x1, x2 = self.prep_data(self.parent.state, self.parent.turn)
        return F.softmax(self.agent.policy(x1, x2), dim=1)[0][move_idx]

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
        # add brackets for [batch, input channel] dimensions
        x1 = torch.tensor([[state]], dtype=torch.float32)
        x2 = torch.tensor([[turn]], dtype=torch.float32)
        return x1, x2
