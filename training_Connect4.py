from pathlib import Path
from agents.Agent_Connect4_MCTS_NN import Agent_Connect4_MCTS_NN
from utils import plot_performance, evaluate_agent
import torch
import torch.nn.functional as F
import random
import pickle
random.seed(20)

output_dir = Path('training_output/Connect4')
n_jobs = 5  # number of processes to use for parallelization
train_games = 1000  # number of games to play in each iteration
eval_games = 100  # number of games to play in each evaluation

def log_stats(p):
    X1 = torch.FloatTensor([[[[-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1, -1, -1]]]])
    X2 = torch.FloatTensor([[0]])
    p.logger.info(f"loss, tie, win prob: {F.softmax(p.value(X1, X2), dim=1)[0]}")
    p.logger.info(f"first move probs: {F.softmax(p.policy(X1, X2), dim=1)[0]}")

def train_and_evaluate():

    # make output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # initialize agent and opponents list
    p = Agent_Connect4_MCTS_NN(simulations=100, depth=7)
    p.setup_logger(output_dir/'log.txt')
    ops = [p.deepcopy_without_data()]
    # ops = [Agent_Connect4_Random(0)]

    # initial model evaluation & stat
    performances = []
    per = evaluate_agent(p, n=eval_games, n_jobs=n_jobs)
    performances.append(per)
    log_stats(p)

    # train agent
    iterations = 15
    for i in range(iterations):
        p.logger.info(f"Iteration {i+1} of {iterations}")
        
        # play vs opponent agents and generate data
        p.gen_data_diff_ops(n=train_games, ops=ops, datapoints_per_game=3, n_jobs=n_jobs)
        
        # train value and policy networks
        p.fit(test_size=0.1, refit_datapoints=200000, early_stopping=50,
            num_epochs=300, verbose=10)
        
        # append to opponents list
        ops.append(p.deepcopy_without_data())
        
        # evaluate against random move agent and mcts agent
        per = evaluate_agent(p, n=eval_games, n_jobs=n_jobs)
        performances.append(per)
        log_stats(p)

    # evaluate vs random move agent and mcts agent with tau=inf
    p.logger.info('Evaluate agent with tau=inf')
    tau = p.tau
    p.tau = None
    evaluate_agent(p, n=eval_games, n_jobs=n_jobs)
    p.tau = tau

    # plot performance
    plot_performance(performances, output_dir)

    # save agents
    p.to_pickle(f"{output_dir}/agent.pkl")
    with open(f"{output_dir}/agent_versions.pkl", 'wb') as f:
        pickle.dump(ops, f)


if __name__ == "__main__":
    train_and_evaluate()
