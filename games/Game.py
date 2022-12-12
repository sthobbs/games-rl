import abc
import random
import itertools
from copy import deepcopy


class Game():
    
    def __init__(self, state, agents, turn=0, store_states=False):
        """
        state: game state (e.g. 3x3 array)
        agents: list of Agents playing the game
        turn: index (starting at 0) of agent whose turn it is
        store_states: if True, keep a sequence of all (state, turn, last_move)'s
            where turn is the player who just played).
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
        """play agent's turn (i.e. update the game state and turn attributes)."""
        self.state, self.last_move = self.agents[self.turn].play_turn(self.state)

    def update_turn(self):
        """update turn to the next player/agent"""
        self.turn = (self.turn + 1) % len(self.agents)

    @abc.abstractmethod
    def result(self):
        """
        Detmine the winner of the game:
            Return None the game is not over.
            Return -1 if the game is over and there is no winner (i.e. a tie).
            Return agent index (i.e. `turn`) of winning player if a player has won.
        """
        ...

    def play_game(self, pprint=True):
        assert self.result() is None, 'no valid moves at beginning of game'
        # store initial state
        if self.store_states:
            self.states.append((deepcopy(self.state), None, None))
        while True:
            # play turn
            self.play_turn()
            # store list of (state, turn, last_move) tuples (where turn is the player who just played)
            if self.store_states:
                self.states.append((deepcopy(self.state), self.turn, self.last_move))
            result = self.result()
            # print result and game state
            if pprint:
                print(result)
                try:
                    self.pretty_print()
                except:
                    print(self.state)
            # check if game is over
            if result is not None:
                break
            # update whos turn it is
            self.update_turn()
        self.winner = result

    @abc.abstractmethod
    def pretty_print(self):
        """print game state in human-readable format."""

    def get_data_point(self, player=None):
        """
        grab a random game state, and output the tuple (state, turn, move, winner)
            `state` is the current game state
            `turn` is the index of the agent that plays next
            `move` is the move the next player will make, (None if it's the terminal game state)
            `winner` is the index of the agent that won, or -1 if the game was a tie.
        this can be used to generate training data.
        `player` can be set to the index of the agent from whom you want to pick a move from
        """
        if player is not None:
            idx = random.randrange(player, len(self.states), len(self.agents))
        else:
            idx = random.randrange(len(self.states))
        state = self.states[idx][0]
        if idx == len(self.states) - 1: # final state of game, so no next move
            turn = self.states[idx-1][1]
            move = None
        else:
            _, turn, move = self.states[idx+1]
        return state, turn, move, self.winner

    def flatten_state(self, replace={' ': -1}):
        """flatten state into a 1d list, replace ' 's"""
        state = list(itertools.chain.from_iterable(self.state)) # flatten 2d list
        for i in range(len(state)):
            if state[i] in replace.keys():
                state[i] = replace[state[i]]
        return state
