from agents.Agent_TicTacToe import Agent_TicTacToe
from mcts.MCTS_NN_Node import MCTS_NN_Node
from mcts.MCTS_TicTacToe_methods import MCTS_TicTacToe_methods
from games.TicTacToe import TicTacToe
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from copy import deepcopy
import random
from tqdm import tqdm

# initialization data (from random games) to explicitly specify classes
X = [[1, 0, -1, -1, 1, 0, -1, -1, 0, 1], [-1, 0, -1, 0, -1, -1, 1, -1, 1, 0], [-1, -1, -1, 0, -1, -1, -1, -1, -1, 1], [-1, -1, -1, 0, -1, 1, -1, 1, 0, 0], [-1, -1, -1, -1, 0, -1, -1, -1, -1, 1], [-1, 0, -1, -1, 1, 1, 0, -1, -1, 0], [-1, 0, 0, -1, 1, 0, -1, -1, 1, 1], [-1, 0, -1, 1, 0, 0, 1, -1, 1, 0], [-1, -1, 1, 0, 0, 1, 1, 0, -1, 0], [0, 0, 1, 0, 1, 0, -1, -1, 1, 1]]
Yv = [1, -1, -1, 1, -1, -1, 1, -1, 1, -1]
Yp = [6, 2, 0, 2, 1, 0, 0, 2, 1, 7]
value = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(100,25), activation='logistic')
value.partial_fit(X, Yv, classes=[-1,0,1])
y_pred = value.predict_proba(X)
y_true = Yv
log_loss(y_true, y_pred, labels=[-1,0,1])

class Agent_TicTacToe_MCTS_NN(Agent_TicTacToe):
    """Agent that plays tic-tac-toe moves based on Monte Carlo Tree Search as well as 2 neural networks.
        - a value network that estimates P(win), and
        - a value network that predicts next move
    """
    def __init__(self, agent_idx=None, simulations=1000, verbose=False, value=None, policy=None):
        super().__init__(agent_idx)        
        self.simulations = simulations # number of simulations for MCTS
        self.verbose = verbose
        # TODO: replace sklearn models with pytorch (may also combine into one NN that gives both value and policy outputs)
        # set value network
        if type(value) is sklearn.neural_network._multilayer_perceptron.MLPClassifier:
            self.value = value
        elif value is None:
            self.value = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(100,25), activation='logistic') #, warm_start=True)
            print("initial fit on value network")
            self.value.partial_fit(X, Yv, classes=[-1,0,1])
            self.value.name = 'value' # name model for reference later
        assert hasattr(self, 'value'), 'Invalid value network'
        # set policy network
        if type(policy) is sklearn.neural_network._multilayer_perceptron.MLPClassifier:
            self.policy = policy
        elif policy is None:
            self.policy = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(100,25), activation='logistic') #, warm_start=True)
            print("initial fit on value policy")
            self.policy.partial_fit(X, Yp, classes=range(9))
            self.policy.name = 'policy' # name model for reference later
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
        for X_chunk, y_chunk in self.iter_minibatches(X, y, batch_size=batch_size):
            model.partial_fit(X_chunk, y_chunk, classes=model.classes_)

    def fit_model(self, X_train, y_train, X_test, y_test, model, early_stopping=100, verbose=50, num_epochs=10):
        """fit either value or policy network for num_epochs epochs """ 

        assert early_stopping or num_epochs, 'must set early_stopping or num_epochs'
        # stop if log loss hasn't reached a new low for {early_stopping} epochs, set model to the one with the minimum log loss
        if early_stopping:
            epoch = 0
            min_log_loss_epoch = 0
            min_log_loss_model = deepcopy(model)
            while True:
                # get metrics (log loss)     
                y_pred = model.predict_proba(X_test)
                test_loss = log_loss(y_test, y_pred, labels=model.classes_)
                y_pred = model.predict_proba(X_train)
                train_loss = log_loss(y_train, y_pred, labels=model.classes_)
                if epoch % verbose == 0: # print metrics
                    print(f"[{epoch}] train log loss = {train_loss}, test log loss = {test_loss}")
                # check best log loss so far
                if epoch == 0:
                    min_log_loss = test_loss # lowest log loss so far
                    min_log_loss_train = train_loss # associated train log loss for lowest test log loss so far
                elif test_loss < min_log_loss:
                    min_log_loss = test_loss
                    min_log_loss_train = train_loss
                    min_log_loss_epoch = epoch
                    min_log_loss_model = deepcopy(model)
                # increment epoch
                epoch += 1
                # check if need to stop
                if epoch > min_log_loss_epoch + early_stopping:
                    # copy optimal model to agent
                    assert model.name in ['value', 'policy'], "model name not in ['value', 'policy']"
                    if model.name == 'value':
                        self.value = min_log_loss_model
                    elif model.name == 'policy':
                        self.policy = min_log_loss_model
                    # print optimal loss
                    print("optimal epoch:")
                    print(f"[{min_log_loss_epoch}] train log loss = {min_log_loss_train}, test log loss = {min_log_loss}")
                    return
                # shuffle data
                X_train, y_train = shuffle(X_train, y_train)
                # fit 1 epoch
                self.fit_epoch(X_train, y_train, model)
                # model.partial_fit(X_train, y_train, classes=model.classes_)

        elif num_epochs:
            for i in range(num_epochs):
                # shuffle data
                X_train, y_train = shuffle(X_train, y_train)
                # fit 1 epoch
                self.fit_epoch(X_train, y_train, model)
                # model.partial_fit(X_train, y_train, classes=model.classes_)
                # print metrics
                y_pred = model.predict_proba(X_train)
                y_true = y_train
                train_loss = log_loss(y_true, y_pred, labels=model.classes_)
                y_pred = model.predict_proba(X_test)
                y_true = y_test
                test_loss = log_loss(y_true, y_pred, labels=model.classes_)
                print(f"[{i}] train log loss = {train_loss}, test log loss = {test_loss}")
            return

    def gen_data_diff_ops(self, n, ops, datapoints_per_game=1):
        """play n games against different opponents, randomly selected from ops,
        and generate data from these games for training/testing 
        """
        ops = deepcopy(ops)
        for op in ops:
            op.agent_idx = 1
        Xv, yv, Xp, yp = [], [], [], []
        for i in tqdm(range(n)):
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
        return Xv, yv, Xp, yp

    def play_and_refit(self, n, ops, datapoints_per_game=1, test_size=0.3, early_stopping=None, num_epochs=1):
        """self-play n games to generate training data, then refit NNs
        `datapoints_per_game` generate this many data points per game,
            although the policy network may have slightly less, because if the final game state
            is selected, it has no next move to predict.
        """
        
        # # set oppopent as copy of current agent
        # op = deepcopy(self)
        # op.agent_idx = 1
        # generate training and testing data independently to avoid leakage from multiple datapoints the from same game
        assert 0 < test_size < 1, 'invalid test size'
        n_train = int((1 - test_size) * n)
        n_test = int(test_size * n)
        print('generating training data')
        Xv_train, yv_train, Xp_train, yp_train = self.gen_data_diff_ops(n_train, ops, datapoints_per_game=datapoints_per_game)
        # Xv_train, yv_train, Xp_train, yp_train = self.gen_data(n_train, agents=[self, op], datapoints_per_game=datapoints_per_game)
        print('generating test data')
        Xv_test,  yv_test,  Xp_test,  yp_test  = self.gen_data_diff_ops(n_test, ops, datapoints_per_game=datapoints_per_game)
        # Xv_test,  yv_test,  Xp_test,  yp_test  = self.gen_data(n_test,  agents=[self, op], datapoints_per_game=datapoints_per_game)
        # fit value network
        print('fitting value network')
        self.fit_model(Xv_train, yv_train, Xv_test, yv_test, model=self.value, early_stopping=early_stopping, num_epochs=num_epochs)
        # fit policy network
        print('fitting policy network')
        self.fit_model(Xp_train, yp_train, Xp_test, yp_test, model=self.policy, early_stopping=early_stopping, num_epochs=num_epochs)


    def play_turn(self, state):
        """the Agent plays a turn, and returns the new game state, along with the move played"""
        mcts = TicTacToe_MCTS_NN_Node(agent=self, state=state, turn=self.agent_idx)
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
        state, move = Agent_TicTacToe_MCTS_NN(
            agent_idx=self.turn,
            value=self.agent.value,
            policy=self.agent.policy 
            ).play_move(self.state, move, deepcopy_state=True)
        return state, move

    def value_predict(self, state, turn):
        """Return output of value networks"""
        x = self.prep_data(state, turn)
        return self.agent.value.predict_proba(x)[0]

    def policy_predict(self, state, turn, move):
        """Return output of policy networks (for a specific move index)"""
        move_idx = TicTacToe(agents=[None, None]).move_index(move)
        x = self.prep_data(state, turn)
        return self.agent.policy.predict_proba(x)[0][move_idx]
