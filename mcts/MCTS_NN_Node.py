import abc
import random
from mcts.MCTS_Node import MCTS_Node
from games.Game import Game


class MCTS_NN_Node(MCTS_Node):
    """Node for Monte Carlo Tree Search that uses Neural Networks for action values"""
    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)
        self.val_w_sum = 0 # sum of P(win) value network predictions of terminal nodes
        self.val_t_sum = 0 # sum of P(tie) value network predictions of terminal nodes
        self.val_l_sum = 0 # sum of P(loss) value network predictions of terminal nodes
        self.c = 1.41
        self.tau = 0.5

    def __str__(self):
        self.N = self.parent.n
        Q = (self.val_l_sum - self.val_w_sum) / self.n # (V(l) - V(w)) / n
        U = self.c * self.policy_predict(self.parent.state, self.parent.turn, self.last_move) * self.N**0.5 / (self.n + 1) # c * p * sqrt(N) / (n+1)        
        win_pct = 'N/A' if self.n == 1 else f"{round(100*self.w/(self.n-1), 2)}%"
        return f"""state = {self.state}, move = {self.last_move}, win_rate = {win_pct}, w = {self.w}, t = {self.t}, n = {self.n-1}, action_value = {round(self.action_value(),8)}, turn = {self.turn},
        val_w_sum = {self.val_w_sum}, val_l_sum = {self.val_l_sum}, val_t_sum = {self.val_t_sum},
        [P(loss), P(tie), P(win)] = {self.value_predict(self.state, self.turn)}, P(move) = {self.policy_predict(self.parent.state, self.parent.turn, self.last_move)}
        Q = {Q}, U = {U}
        """ 

    @abc.abstractmethod
    def value_predict(self, state, turn):
        """Return output of value networks"""
        ...

    @abc.abstractmethod
    def policy_predict(self, state, turn, move):
        """Return output of policy networks (for a specific move index)"""
        ...

    def prep_data(self, state, turn):
        """prepare input for policy or value network (it's the same input)"""
        # state, move = TicTacToe_Random(self.turn).play_move(self.state, move, deepcopy_state=True)
        state = Game(state, agents=[None]).flatten_state() # flatten state and replace ' ' with -1
        x = state + [turn]
        return [x]

    def action_value(self):
        """get action value for current node."""
        self.N = self.parent.n
        Q = (self.val_l_sum - self.val_w_sum) / self.n # L - W since it's from the opponent's perspective
        U = self.c * self.policy_predict(self.parent.state, self.parent.turn, self.last_move) * self.N**0.5 / (self.n + 1) # c * p * sqrt(N) / (n+1)
        return Q + U

    def traverse_down(self):
        """traverse down the tree, picking optimal child nodes."""
        if not self.children: # if children is empty, try setting them
            self.set_children()
        result = self.game_result()
        assert self.children or result is not None, "game isn't over, but there are no child nodes"
        if result is not None: # then the game is over (i.e. I'm at a leaf), so go back up the tree updating values
            self.winner = result
            # swap order on win and loss since it's from the perspective of the next player (i.e. the one that didn't just end the game)
            self.val_w, self.val_t, self.val_l = self.value_predict(self.state, self.turn) # value network prediction of leaf (or at max depth to be implemented) of simulation
            self.traverse_up()
            return
        if self.n <= 5: # pick random child for the first x=5 simulations (so MCTS isn't deterministic)
            child = self.children[random.randrange(len(self.children))] # get random child
        else: # there are children:
            child = self.max_action_child() # find optimal child based on action values
        child.traverse_down()

    def traverse_up(self):
        """traverse up the tree, updating counts."""
        if self.winner == self.turn:
            self.w += 1 # increment num wins
        elif self.winner == -1:
            self.t += 1 # increment number of ties
        self.val_w_sum += self.val_w # accumulate sum of value network (wins)
        self.val_t_sum += self.val_t # accumulate sum of value network (ties)
        self.val_l_sum += self.val_l # accumulate sum of value network (loss)
        self.n += 1 # increment num visits    
        if self.parent is not None:
            self.parent.winner = self.winner # propagate winner up
            self.parent.val_w = self.val_l # propagate terminal value network output (have to alternate win and loss since the turn is alternating)
            self.parent.val_t = self.val_t
            self.parent.val_l = self.val_w
            self.parent.traverse_up()
