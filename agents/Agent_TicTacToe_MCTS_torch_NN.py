from agents.Agent_TicTacToe import Agent_TicTacToe
from mcts.MCTS_NN_Node import MCTS_NN_Node
from mcts.MCTS_TicTacToe_methods import MCTS_TicTacToe_methods
from games.TicTacToe import TicTacToe
from games.Game import Game
from copy import deepcopy
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

criterion = nn.CrossEntropyLoss()


class Value(nn.Module):
    """Value network for agent."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 120)
        self.fc2 = nn.Linear(120, 25)
        self.fc3 = nn.Linear(25, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc3(x), dim=1)
        x = self.fc3(x)
        return x

class Policy(nn.Module):
    """Policy network for agent."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 120)
        self.fc2 = nn.Linear(120, 25)
        self.fc3 = nn.Linear(25, 9)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc3(x), dim=1)
        x = self.fc3(x)
        return x



class Agent_TicTacToe_MCTS_torch_NN(Agent_TicTacToe):
    """Agent that plays tic-tac-toe moves based on Monte Carlo Tree Search as well as 2 neural networks.
        - a value network that estimates P(win), and
        - a value network that predicts next move
    """
    def __init__(self, agent_idx=None, simulations=1000, depth=None, verbose=False, value=None, policy=None):
        super().__init__(agent_idx)        
        self.simulations = simulations # number of simulations for MCTS
        self.depth = depth
        self.verbose = verbose
        # set value network
        if value is None:
            self.value = Value() 
            self.value.name = 'value' # name model for reference later
        else:
            self.value = value
        assert hasattr(self, 'value'), 'Invalid value network'
        # set policy network
        if policy is None:
            self.policy = Policy()
            self.policy.name = 'policy' # name model for reference later
        else:
            self.policy = policy
        assert hasattr(self, 'policy'), 'Invalid policy network'

    def iter_minibatches(self, X, y, batch_size=32):
        # Provide chunks one by one
        cur = 0
        while cur < len(y):
            chunkrows = slice(cur, cur + batch_size)
            X_chunk, y_chunk = X[chunkrows], y[chunkrows]
            yield X_chunk, y_chunk
            cur += batch_size

    def fit_epoch(self, X, y, model, batch_size=32):
        """fit one epoch"""

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # randomly permute data
        idx = torch.randperm(X.shape[0])
        X = X[idx].view(X.size())
        y = y[idx].view(y.size())

        for X_chunk, y_chunk in self.iter_minibatches(X, y, batch_size=batch_size):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(X_chunk)
            # breakpoint()
            loss = criterion(outputs, y_chunk)
            loss.backward()
            optimizer.step()

    def fit_model(self, X_train, y_train, X_test, y_test, model, early_stopping=None, verbose=10, num_epochs=10):
        """fit either value or policy network for num_epochs epochs.""" 
        if early_stopping is None:
            early_stopping = float("inf")

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            # shuffle data
            # X_train, y_train = shuffle(X_train, y_train)
            # fit 1 epoch
            self.fit_epoch(X_train, y_train, model)
            # print metrics
            y_pred = model(X_train)
            y_true = y_train
            train_loss = criterion(y_pred, y_true).item()
            y_pred = model(X_test)
            y_true = y_test
            test_loss = criterion(y_pred, y_true).item()
            if epoch % verbose == 0: # print metrics
                print(f"[{epoch}] train log loss = {train_loss}, test log loss = {test_loss}")
            # check best log loss so far
            if epoch == 0:
                min_log_loss = test_loss # lowest log loss so far
                min_log_loss_train = train_loss # associated train log loss for lowest test log loss so far
                min_log_loss_epoch = 0 # epoch with lowest test loss
                min_log_loss_model = deepcopy(model) # model with lowest test loss
            elif test_loss < min_log_loss:
                min_log_loss = test_loss
                min_log_loss_train = train_loss
                min_log_loss_epoch = epoch
                min_log_loss_model = deepcopy(model)
            # early stopping
            if epoch > min_log_loss_epoch + early_stopping:
                break
        print("optimal epoch:")
        print(f"[{min_log_loss_epoch}] train log loss = {min_log_loss_train}, test log loss = {min_log_loss}")              
        # copy optimal model to agent
        assert model.name in ['value', 'policy'], "model name not in ['value', 'policy']"
        if model.name == 'value':
            self.value = min_log_loss_model
        elif model.name == 'policy':
            self.policy = min_log_loss_model
        return

    def gen_data_diff_ops(self, n, ops, datapoints_per_game=1):
        """play n games against different opponents, randomly selected from ops,
        and generate data from these games for training/testing 
        """
        # deepcopy agents so the original agent_idx's aren't changed
        # ops = deepcopy(ops)
        # cur = deepcopy(self)
        Xv, yv, Xp, yp = [], [], [], []
        for _ in tqdm(range(n)):
            op = random.choice(ops) # pick random opponent
            # randomly pick who goes first
            if random.random() < 0.5:
                agents = [self, op]
                player = 0
            else:
                agents = [op, self]
                player = 1
            Xv_, yv_, Xp_, yp_ = self.gen_data(1, agents=agents, player=player, datapoints_per_game=datapoints_per_game, verbose=False)
            Xv.extend(Xv_)
            yv.extend(yv_)
            Xp.extend(Xp_)
            yp.extend(yp_)
        # augment data
        Xv, yv, Xp, yp = self.augment_data(Xv, yv, Xp, yp)
        # convert to tensors
        Xv = torch.FloatTensor(Xv)
        yv = torch.FloatTensor(yv)
        Xp = torch.FloatTensor(Xp)
        yp = torch.LongTensor(yp)
        return Xv, yv, Xp, yp

    def augment_data(self, Xv, yv, Xp, yp):
        """since tic tac toe is invariant to rotations and reflections,
        return all rotations and reflections of data.
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
        """rotate Xv (or Xp) and yp right 90 degrees"""
        X_ = [X[6], X[3], X[0], X[7], X[4], X[1], X[8], X[5], X[2], X[9]]
        y_ = [2, 5, 8, 1, 4, 7, 0, 3, 6][y]
        return X_, y_
    
    def reflect(self, X, y=0):
        """reflect Xv (or Xp) and yp along the veritcal line of symmetry"""
        X_ = [X[2], X[1], X[0], X[5], X[4], X[3], X[8], X[7], X[6], X[9]]
        y_ = [2, 1, 0, 5, 4, 3, 8, 7, 6][y]
        return X_, y_

    def play_and_refit(self, n, ops, datapoints_per_game=1, test_size=0.3, **kwargs):
        """self-play n games to generate training data, then refit NNs
        `datapoints_per_game` generate this many data points per game,
            although the policy network may have slightly less, because if the final game state
            is selected, it has no next move to predict.
        """
        
        # generate training and testing data from different games to avoid leakage
        assert 0 < test_size < 1, 'invalid test size'
        n_train = int((1 - test_size) * n)
        n_test = int(test_size * n)
        print('generating training data')
        Xv_train, yv_train, Xp_train, yp_train = self.gen_data_diff_ops(n_train, ops, datapoints_per_game=datapoints_per_game)
        print('generating test data')
        Xv_test,  yv_test,  Xp_test,  yp_test  = self.gen_data_diff_ops(n_test, ops, datapoints_per_game=datapoints_per_game)
        # fit value network
        print('fitting value network')
        self.fit_model(Xv_train, yv_train, Xv_test, yv_test, model=self.value, **kwargs)
        # fit policy network
        print('fitting policy network')
        self.fit_model(Xp_train, yp_train, Xp_test, yp_test, model=self.policy, **kwargs)

    def play_turn(self, state):
        """the Agent plays a turn, and returns the new game state, along with the move played"""
        mcts = TicTacToe_MCTS_NN_Node(agent=self, state=state, turn=self.agent_idx, depth=self.depth)
        mcts.simulations(self.simulations) # play simulations
        move = mcts.best_move(verbose=self.verbose) # find best most
        state, move = self.play_move(state, move) # play move
        return state, move


class TicTacToe_MCTS_NN_Node(MCTS_TicTacToe_methods, MCTS_NN_Node):
    """Monte Carlo Tree Search Node for a game of tic tac toe"""

    def __init__(self, agent, *args, **kwargs):
        super().__init__(agent=agent, *args, **kwargs)
        self.agent = agent

    def play_move(self, move):
        """play a specific move, returning the new game state (and the move played)."""
        state, move = Agent_TicTacToe_MCTS_torch_NN(
            agent_idx=self.turn,
            value=self.agent.value,
            policy=self.agent.policy 
            ).play_move(self.state, move, deepcopy_state=True)
        return state, move

    def value_predict(self):
        """Return output of value networks"""
        x = self.prep_data(self.state, self.turn)
        return F.softmax(self.agent.value(x), dim=1)[0]

    def policy_predict(self):
        """Return output of policy networks for the parent's state,
        parent's turn, and last move played"""
        move_idx = TicTacToe.move_index(self.last_move)
        x = self.prep_data(self.parent.state, self.parent.turn)
        return F.softmax(self.agent.policy(x), dim=1)[0][move_idx]

    def prep_data(self, state, turn):
        """prepare input for policy or value network (it's the same input)"""
        state = Game.replace_2d(state)
        state = Game.flatten_state(state)
        x = state + [turn]
        x = torch.FloatTensor([x])
        return x