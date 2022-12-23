import abc
import random
import itertools
from copy import deepcopy


class Game():
    """Class for a generic game."""

    def __init__(self, state, agents, turn=0, store_states=False):
        """
        Initialize a game.

        Parameters
        ----------
        state : any
            Game state (e.g. 3x3 array for tic tac toe)
        agents : list
            List of agents playing the game
        turn : int
            Index (starting at 0) of agent whose turn it is
        store_states : bool
            If True, keep a sequence of all (state, turn, last_move) tuples
                Where turn is the player who just played.
                This can be used to generate training data.
        """
        self.state = state
        self.agents = agents
        assert turn in list(range(len(agents))), f'invalid turn: {turn}'
        self.turn = turn
        self.store_states = store_states
        if store_states:
            self.states = []
        self.last_move = None

    def play_turn(self):
        """Play agent's turn (i.e. update the game state and last move attributes)."""
        self.state, self.last_move = self.agents[self.turn].play_turn(self.state)

    def update_turn(self):
        """Update turn to the next player/agent."""
        self.turn = (self.turn + 1) % len(self.agents)

    @abc.abstractmethod
    def result(self):
        """
        Determine the winner of the game.

        Return
        ------
        None if the game is not over.
        -1 if the game is over and there is no winner (i.e. a tie).
        agent index (i.e. `turn`) of winning player if a player has won.
        """

    def play_game(self, pprint=True):
        """
        Play a game.

        Parameters
        ----------
        pprint : bool
            If True, print the game state after each turn. Use for human vs bot play.
        """
        assert self.result() is None, 'no valid moves at beginning of game'
        # store initial state
        if self.store_states:
            self.states.append((deepcopy(self.state), None, None))
        while True:
            # play turn
            self.play_turn()
            # store list of (state, turn, last_move) tuples, where turn is the player
            # who just played
            if self.store_states:
                self.states.append((deepcopy(self.state), self.turn, self.last_move))
            result = self.result()
            # print game state
            if pprint:
                try:
                    self.pretty_print()
                except NotImplementedError:
                    print(self.state)
            # check if game is over
            if result is not None:
                # print game result
                if pprint:
                    if result == -1:
                        print("tie game")
                    else:
                        print(f"player {result} wins")
                break
            # update whos turn it is
            self.update_turn()
        self.winner = result

    @abc.abstractmethod
    def pretty_print(self):
        """Print game state in human-readable format."""

    def get_data_point(self, player=None):
        """
        Construct data point from a random saved state. This can be used
        to generate training data.

        Parameters
        ----------
        player : int or None
            Can be set to the index of the agent to pick a random move from.
            If None, pick a random state from any player

        Returns
        -------
        state : list
            Game state (e.g. 3x3 array for tic tac toe)
        turn : int
            Index (starting at 0) of agent whose turn it is
        move : tuple
            Move the next player will make, (None if it's the terminal game state)
        winner : int
            Index of the agent that won, or -1 if the game was a tie.
        """
        # pick a random state (from any player)
        if player is None:
            idx = random.randrange(len(self.states))
        # pick a random state from a specific player
        else:
            idx = random.randrange(player, len(self.states), len(self.agents))
        state = self.states[idx][0]
        # if final state of game, there is no next move
        if idx == len(self.states) - 1:
            turn = self.states[idx-1][1]
            move = None
        else:
            _, turn, move = self.states[idx+1]
        return state, turn, move, self.winner

    @staticmethod
    def flatten_state(state):
        """
        Flatten state into a 1d list.

        Parameters
        ----------
        state : list
            Game state (e.g. 3x3 array for tic tac toe)
        """
        state = list(itertools.chain.from_iterable(state))  # flatten 2d list
        return state

    @staticmethod
    def replace_2d(state, replace={' ': -1}, copy=True):
        """
        Replace values in 2d list.

        Parameters
        ----------
        state : list
            Game state (e.g. 3x3 array for tic tac toe)
        replace : dict
            Dictionary of values to replace (e.g. {' ': -1} to replace spaces with -1)
        copy : bool
            If True, make a copy of the state before replacing values.
        """
        if copy:
            state = deepcopy(state)
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] in replace.keys():
                    state[i][j] = replace[state[i][j]]
        return state
