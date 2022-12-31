from agents.Agent_TicTacToe import Agent_TicTacToe
from agents.Agent_TicTacToe_Random import Agent_TicTacToe_Random
from agents.Agent_TicTacToe_MCTS import Agent_TicTacToe_MCTS
from agents.Agent_Connect4 import Agent_Connect4
from agents.Agent_Connect4_Random import Agent_Connect4_Random
from agents.Agent_Connect4_MCTS import Agent_Connect4_MCTS
import matplotlib.pyplot as plt


def plot_performance(performances, output_dir, games_per_iteration=1000):
    """
    Plot loss rates for an agent against random move and monte carlo tree search agents.
    
    Parameters
    ----------
    performances : list
        list of performance lists
    """
    # get loss rates
    rand_loss_rate1 = [v[0][0] / sum(v[0]) for v in performances]
    rand_loss_rate2 = [v[1][0] / sum(v[0]) for v in performances]
    mcts_loss_rate1 = [v[2][0] / sum(v[0]) for v in performances]
    mcts_loss_rate2 = [v[3][0] / sum(v[0]) for v in performances]
    training_games = [games_per_iteration * i for i in range(len(performances))]
    # plot loss rates
    plt.figure()
    with plt.style.context('seaborn-darkgrid'):
        plt.plot(training_games, rand_loss_rate1, label='going 1st vs random')
        plt.plot(training_games, rand_loss_rate2, label='going 2nd vs random')
        plt.plot(training_games, mcts_loss_rate1, label='going 1st vs mcts')
        plt.plot(training_games, mcts_loss_rate2, label='going 2nd vs mcts')
        plt.title(f'Loss Rates During Training')
        plt.xlabel('Number of Training Games')
        plt.ylabel('Loss Rate')
        plt.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right', borderaxespad=0, frameon=True)
        plt.savefig(f'{output_dir}/loss_rates.png', bbox_inches='tight', pad_inches=0.3)
    plt.close()


def evaluate_agent(p, n=100, n_jobs=1):
    """
    Evaluate agent against a random move agent and a mcts agent.
    
    Parameters
    ----------
    p : Agent_TicTacToe or Agent_Connect4
        agent to evaluate
    n : int
        number of games to play per opponent
    """
    # initialize random move agent and mcts agent
    if isinstance(p, Agent_TicTacToe):
        r = Agent_TicTacToe_Random()
        m = Agent_TicTacToe_MCTS(simulations=100)
    elif isinstance(p, Agent_Connect4):
        r = Agent_Connect4_Random()
        m = Agent_Connect4_MCTS(simulations=100)
    else:
        raise ValueError(f'Agent type not recognized.')
    # play games and get performance (loss, tie, win counts)
    p.logger.info(f'Evaluating Agent:')
    p.logger.info(f'going 1st vs random')
    per1 = p.gen_data(n, agents=[p, r], return_results=True, n_jobs=n_jobs)  # going 1st vs random
    p.logger.info(f'going 2nd vs random')
    per2 = p.gen_data(n, agents=[r, p], return_results=True, n_jobs=n_jobs)  # going 2nd vs random
    p.logger.info(f'going 1st vs mcts')
    per3 = p.gen_data(n, agents=[p, m], return_results=True, n_jobs=n_jobs)  # going 1st vs mcts
    p.logger.info(f'going 2nd vs mcts')
    per4 = p.gen_data(n, agents=[m, p], return_results=True, n_jobs=n_jobs)  # going 2nd vs mcts
    performance = [per1, per2, per3, per4]
    return performance
