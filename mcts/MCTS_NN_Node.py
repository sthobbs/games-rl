import abc
import random
from mcts.MCTS_Node import MCTS_Node


class MCTS_NN_Node(MCTS_Node):
    """Node for Monte Carlo Tree Search that uses Neural Networks to guide the search."""
    
    def __init__(self, state, depth=None, *args, **kwargs):
        """
        Initialize node.
        
        Parameters
        ----------
        state : any
            Game state representation
        depth : int
            Search depth for MCTS
        """
        super().__init__(state, *args, **kwargs)
        self.val_w_sum = 0 # sum of P(win) value network predictions of terminal nodes
        self.val_t_sum = 0 # sum of P(tie) value network predictions of terminal nodes
        self.val_l_sum = 0 # sum of P(loss) value network predictions of terminal nodes
        self.c = 1.41
        self.tau = 0.5
        self.depth = depth # search depth for MCTS
        self.winner = None # index of agent who won the game (-1 => tie, None => game not over)

    def __str__(self):
        """MCTS Node details, primarily for debugging."""
        self.N = self.parent.n
        Q = (self.val_l_sum - self.val_w_sum) / self.n # (V(l) - V(w)) / n
        U = self.c * self.policy_predict() * self.N**0.5 / (self.n + 1) # c * p * sqrt(N) / (n+1)        
        win_pct = 'N/A' if self.n == 1 else f"{round(100*self.w/(self.n-1), 2)}%"
        return f"""state = {self.state}, move = {self.last_move}, win_rate = {win_pct}, w = {self.w}, t = {self.t}, n = {self.n-1}, action_value = {self.action_value()}, turn = {self.turn},
        val_w_sum = {self.val_w_sum}, val_l_sum = {self.val_l_sum}, val_t_sum = {self.val_t_sum},
        [P(loss), P(tie), P(win)] = {self.value_predict()}, P(move) = {self.policy_predict()}
        Q = {Q}, U = {U}
        """ 

    @abc.abstractmethod
    def value_predict(self):
        """Return output of value networks based on the current state and turn"""

    @abc.abstractmethod
    def policy_predict(self):
        """Return output of policy networks for the parent's state,
        parent's turn, and last move played"""

    def action_value(self):
        """Get action value for current node."""
        self.N = self.parent.n
        Q = (self.val_l_sum - self.val_w_sum) / self.n # L - W since it's from the opponent's perspective
        U = self.c * self.policy_predict() * self.N**0.5 / (self.n + 1) # c * p * sqrt(N) / (n+1)
        return Q + U

    def traverse_down(self):
        """Traverse down the tree, picking optimal child nodes."""
        # if children is empty, try setting them
        if not self.children:
            depth = None if self.depth is None else self.depth - 1
            self.set_children(depth=depth)
        # get game status (including if game is not over)
        result = self.game_result()
        assert self.children or result is not None, "game isn't over, but there are no child nodes"
        # if game is over, set winner
        if result is not None:
            self.winner = result
        # if game is over or at max depth, update values and go back up the tree
        if result is not None or self.depth == 1:
            # get value network predictions of final position in simulation
            self.val_l, self.val_t, self.val_w = self.value_predict()
            self.traverse_up()
            return
        # pick random child for the first 10 simulations
        if self.n <= 10:
            child = random.choice(self.children)
        # find optimal child based on action values
        else:
            child = self.max_action_child()
        # recursively traverse down the tree
        child.traverse_down()

    def traverse_up(self):
        """Traverse up the tree, updating counts."""
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
