import math
import abc
import random


class MCTS_Node():
    """Monte Carlo Tree Search Node"""

    def __init__(self, state, parent=None, turn=None, last_move=None, tau=0.25, *args, **kwargs):
        self.state = state # game state representation 
        self.parent = parent # Node's parent (the root's parent is None) 
        self.children = [] # Node's children, to be determined
        self.w = 0 # num wins from current node
        self.t = 0 # num ties from current node
        self.n = 1 # num visits to current node
        self.c = 1.41 # hyperparameter, sqrt(2) is common for simple MCTS
        self.last_move = last_move # most recent move that updated the game state
        self.tau = tau # tau > 0. pick move with probability p**(1/tau)
        if parent is None: # if root
            assert turn is not None, 'turn must be specified for the root'
            self.turn = turn
            self.N = None
        else:
            self.turn = parent.turn
            self.update_turn()
            self.N = parent.n # num visits to parent node
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        win_rate = round(100*self.w/(self.n-1), 2)
        action_value = round(self.action_value(),8)
        return f"state = {self.state}, move = {self.last_move}, win_rate = {win_rate}%, \
            w = {self.w}, t = {self.t}, n = {self.n-1}, action_value = {action_value}, turn = {self.turn}"

    @abc.abstractmethod
    def valid_moves(self):
        """return a list of valid next moves for a given game state"""
        ...

    @abc.abstractmethod
    def game_result(self):
        """
        Detmine the winner of the game:
            Return agent index of winning player if a player has won.
            Return None the game is not over.
            Return -1 if the game is over and there is no winner (i.e. a tie).
        """
        ...
    
    @abc.abstractmethod
    def play_move(self, move):
        """play a specific move, returning the new game state and the move played."""
        ...
    
    def simulations(self, n):
        """run MCTS with n simulations"""
        for _ in range(n):
            self.traverse_down()

    def best_move(self, verbose=False):
        """return move based on which child node has the most visits n."""
        if self.tau: # randomly pick child with probabilities proportional to n**(1/tau)
            weights = [x.n ** (1 / self.tau) for x in self.children]
            child = random.choices(self.children, weights=weights)[0]
        else: # get child node with max number of visits n
            child = max(self.children, key=lambda x: x.n)
        move = child.last_move
        if verbose: # for debugging individual games
            print(f"Player {self.turn}: {move}")
            self.print_children()
            print()
        return move   

    def print_children(self):
        """print info on child nodes"""
        for child in self.children:
            print(child)

    def set_children(self, **kwargs):
        """instantiate child nodes for each possible valid next move"""
        moves = self.valid_moves()
        for move in moves:
            new_state, _ = self.play_move(move)
            self.children.append(self.__class__(state=new_state, parent=self, last_move=move, *self.args, **self.kwargs, **kwargs))

    def action_value(self):
        """get action value for current node."""
        self.N = self.parent.n
        # (L-W)/n + c*sqrt(ln(N)/n), # L-W becuase it's from the opponent's perspective
        return ((self.n - 1 - self.w - self.t) - self.w) / self.n + self.c * (math.log(self.N) / self.n)**0.5

    def max_action_child(self):
        """return child node with max action_value."""
        max_child = max(self.children, key=lambda x: x.action_value())
        return max_child

    def traverse_down(self):
        """traverse down the tree, picking optimal child nodes."""
        if not self.children: # if children is empty, try setting them
            self.set_children()
        result = self.game_result()
        assert self.children or result is not None, "game isn't over, but there are no child nodes"
        if result is not None: # then the game is over (i.e. I'm at a leaf), so go back up the tree updating values
            self.winner = result
            self.traverse_up()
            return
        if self.n <= 10: # pick random child for the first x=5 simulations (so MCTS isn't deterministic)
            child = self.children[random.randrange(len(self.children))] # get random child
        else:
            child = self.max_action_child() # find optimal child based on action values
        child.traverse_down()

    def traverse_up(self):
        """traverse up the tree, updating counts."""
        if self.winner == self.turn:
            self.w += 1 # increment num wins
        elif self.winner == -1:
            self.t += 1 # increment number of ties
        self.n += 1 # increment num visits    
        if self.parent is not None:
            self.parent.winner = self.winner # propagate winner up
            self.parent.traverse_up()

    def update_turn(self, num_players=2):
        """update turn to the next player/agent"""
        self.turn = (self.turn + 1) % num_players
