# This script trains an agent to play Connect4 using MCTS guided by neural networks.

from pathlib import Path
from agents.Agent_Connect4_MCTS_NN import Agent_Connect4_MCTS_NN
from utils import plot_performance, evaluate_agent
import torch
import torch.nn.functional as F
import random
import pickle
import numpy as np

# if you get "IOError: [Errno 24] Too many open files:", then run "ulimit -n 50000"
# in terminal to increase the limit for number of open files.

# Since we're repeating single inference, overhead from more threads slows down the
# process a lot. This is very important for models with Conv2d layers on machines with
# many cores.
torch.set_num_threads(1)

# set seed
seed = 20
random.seed(seed)
np.random.seed(seed)

output_dir = Path('training_output/Connect4_1')
n_jobs = 112  # number of processes to use for parallelization (set to number of vCPUs)
train_games = 5000  # number of games to play in each iteration
eval_games = 100  # number of games to play in each evaluation
iterations = 12  # number of iterations to train for
refit_datapoints = 400000  # number of datapoints to use for refitting the model
early_stopping = 10  # number of epochs to wait before early stopping


def log_stats(p):
    X1 = torch.FloatTensor([[[[-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1]]]])
    X2 = torch.FloatTensor([[0]])
    p.logger.info("1st move stats:")
    p.logger.info(f"loss, tie, win prob: {F.softmax(p.value(X1, X2), dim=1)[0]}")
    p.logger.info(f"move probs: {F.softmax(p.policy(X1, X2), dim=1)[0]}")
    X1 = torch.FloatTensor([[[[-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1,  0, -1, -1, -1]]]])
    X2 = torch.FloatTensor([[1]])
    p.logger.info("2nd move stats:")
    p.logger.info(f"loss, tie, win prob: {F.softmax(p.value(X1, X2), dim=1)[0]}")
    p.logger.info(f"move probs: {F.softmax(p.policy(X1, X2), dim=1)[0]}")


def train_and_evaluate():

    # make output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # initialize agent and opponents list
    p = Agent_Connect4_MCTS_NN(simulations=100, depth=5)
    p.setup_logger(output_dir/'log.txt')
    ops = [p.deepcopy_without_data()]

    # initial model evaluation & stat
    performances = []
    per = evaluate_agent(p, n=eval_games, n_jobs=n_jobs)
    performances.append(per)
    log_stats(p)

    # train agent
    for i in range(iterations):
        p.logger.info(f"Iteration {i+1} of {iterations}")

        # play vs opponent agents and generate data
        p.gen_data_diff_ops(n=train_games, ops=ops, datapoints_per_game=12, n_jobs=n_jobs)

        # train value and policy networks
        p.fit(test_size=0.1, refit_datapoints=refit_datapoints, early_stopping=early_stopping,
              num_epochs=300, verbose=10)

        # append to opponents list
        ops.append(p.deepcopy_without_data())

        # evaluate against random move agent and mcts agent
        per = evaluate_agent(p, n=eval_games, n_jobs=n_jobs)
        performances.append(per)
        log_stats(p)

        # plot performance
        plot_performance(performances, output_dir, games_per_iteration=train_games)

        # save agents and performances after each iteration
        p.to_pickle(f"{output_dir}/agent.pkl")
        with open(f"{output_dir}/agent_versions.pkl", 'wb') as f:
            pickle.dump(ops, f)

        with open(f"{output_dir}/performances.pkl", 'wb') as f:
            pickle.dump(performances, f)

    # # evaluate vs random move agent and mcts agent with tau=inf
    # p.logger.info('Evaluate agent with tau=inf')
    # tau = p.tau
    # p.tau = None
    # evaluate_agent(p, n=eval_games, n_jobs=n_jobs)
    # p.tau = tau


if __name__ == "__main__":
    train_and_evaluate()
