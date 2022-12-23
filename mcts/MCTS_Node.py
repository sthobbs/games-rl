import math
import abc
import random


class MCTS_Node():
    """Monte Carlo Tree Search Node."""

    def __init__(self, state, parent=None, turn=None, last_move=None, tau=0.25, *args, **kwargs):
        """
        Initialize a MCTS Node.

        Parameters
        ----------
        state : any
            Game state representation
        parent : MCTS_Node
            Node's parent (the root's parent is None)
        turn : int 
            Agent index of current player
        last_move : any
            Most recent move that updated the game state
        tau : float
            tau > 0. pick move with probability p**(1/tau)
        args : tuple
            args to pass to Node's children
        kwargs : dict
            kwargs to pass to Node's children
        """
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
        """MCTS Node details, primarily for debugging."""
        win_rate = round(100*self.w/(self.n-1), 2)
        action_value = round(self.action_value(),8)
        return f"state = {self.state}, move = {self.last_move}, win_rate = {win_rate}%, \
            w = {self.w}, t = {self.t}, n = {self.n-1}, action_value = {action_value}, turn = {self.turn}"

    @abc.abstractmethod
    def valid_moves(self):
        """Return a list of valid next moves for a given game state"""

    @abc.abstractmethod
    def game_result(self):
        """
        Determine the winner of the game.

        Return
        ------
        None if the game is not over.
        -1 if the game is over and there is no winner (i.e. a tie).
        agent index (i.e. `turn`) of winning player if a player has won.
        """
    
    @abc.abstractmethod
    def play_move(self, move):
        """
        play a specific move, returning the new game state.
        
        Parameters
        ----------
        move : any
            move to play

        Returns
        -------
        new_state : any
            new game state after playing move
        """
    
    def simulations(self, n):
        """
        Run MCTS with n simulations.
        
        Parameters
        ----------
        n : int
            number of simulations to run
        """
        for _ in range(n):
            self.traverse_down()

    def best_move(self, verbose=False):
        """
        Return move based on which child node has the most visits n.
        
        Parameters
        ----------
        verbose : bool
            Print info on child nodes. For debugging individual games.
        """
        if self.tau:
            # randomly pick child with probabilities proportional to n**(1/tau)
            weights = [x.n ** (1 / self.tau) for x in self.children]
            child = random.choices(self.children, weights=weights)[0]
        else:
            # get child node with max number of visits n
            child = max(self.children, key=lambda x: x.n)
        move = child.last_move
        if verbose:
            # print info for debugging individual games
            print(f"Player {self.turn}: {move}")
            self.print_children()
            print()
        return move   

    def print_children(self):
        """Print info on child nodes."""
        for child in self.children:
            print(child)

    def set_children(self, **kwargs):
        """Instantiate child nodes for each possible valid next move."""
        moves = self.valid_moves()
        for move in moves:
            new_state = self.play_move(move)
            self.children.append(self.__class__(state=new_state, parent=self, last_move=move, *self.args, **self.kwargs, **kwargs))

    def action_value(self):
        """Get action value for current node."""
        self.N = self.parent.n
        # (L-W)/n + c*sqrt(ln(N)/n), # L-W becuase it's from the opponent's perspective
        return ((self.n - 1 - self.w - self.t) - self.w) / self.n + self.c * (math.log(self.N) / self.n)**0.5

    def max_action_child(self):
        """Return child node with max action_value."""
        max_child = max(self.children, key=lambda x: x.action_value())
        return max_child

    def traverse_down(self):
        """Traverse down the tree, picking optimal child nodes."""
        # if children is empty, try setting them
        if not self.children:
            self.set_children()
        # get game status (including if game is not over)
        result = self.game_result()
        assert self.children or result is not None, "game isn't over, but there are no child nodes"
        # if game is over, go back up the tree updating values
        if result is not None:
            self.winner = result
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
        self.n += 1 # increment num visits    
        # if not root, recursively traverse up the tree
        if self.parent is not None:
            self.parent.winner = self.winner # propagate winner up
            self.parent.traverse_up()

    def update_turn(self, num_players=2):
        """
        Update turn to the next player (i.e. agent)
        
        Parameters
        ----------
        num_players : int
            number of players/agents
        """
        self.turn = (self.turn + 1) % num_players
