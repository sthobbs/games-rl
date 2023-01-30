# AlphaGo-Inspired Reinforcement Learning

DeepMind's [AlphaGo](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf) beating world champion Lee Sedol in the game of Go in 2016 was my initial inspiration to study machine learning. Applying many their ideas here, we use reinforcement learning to trains agents to play Tic Tac Toe and Connect 4 with a modified Monte-Carlo Tree Search (MCTS) algorithm that's guided by neural networks.

From a given game state, a policy network predicts the next move, and a value network predicts the outcome of the game (win, loss, or tie). The training process involves alternating between self-play (including against previous version of the agent to reduce overfitting) to generate training data, and supervised learning of the policy and value networks based on data generated from the self-play.

To evaluate the agent's progress during training, it plays against other agents who (1) pick moves completely randomly, and (2) use traditional Monte Carlo Tree Search.

![image](/training_output/TicTacToe/loss_rates.png?raw=true "Tic Tac Toe Agent Loss Rates")
