# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Austral Intelligence Labs 01. WordRLe - Wordle Environment
------------------------------------------------------------------------------------------------------------------------
Date: 20 of January, 2022
Considerations:
    - Algorithms trained and programmed following different techniques to play the game 'Wordle'

Authors:
        Ignacio Brottier González           ignacio.brottier@accenture.com
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importing libraries and necessary functions
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import random
from gym import Env
from gym.spaces import Discrete, Box, Tuple, MultiBinary, MultiDiscrete
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


class WordleEnv(Env):
    word_list = [row['HEADER'] for _, row in pd.read_csv('five_letter_words.txt').iterrows()]
    letter_dict = {row['IDX']: row['VALUE'] for _, row in pd.read_csv('letters.txt').iterrows()}

    def __init__(self, seed: int = None):

        # Initial State
        self.seed: int = random.randint(0, len(WordleEnv.word_list) - 1) if seed is None else seed
        assert 0 <= self.seed < len(WordleEnv.word_list)
        self.answer: [chr] = [x for x in WordleEnv.word_list[self.seed]]

        self.remaining_turns = 6
        self.turns: [[chr]] = [['', '', '', '', ''] for x in range(6)]
        self.turn_scores: [[int]] = [[0, 0, 0, 0, 0] for x in range(6)]

        self.possible_words: [int] = [x for x in range(len(WordleEnv.word_list))]
        self.possible_letters: [[int]] = [[x for x in range(len(list(WordleEnv.letter_dict.values())))] for y in range(6)]

        self.done = False

        # Action Space
        # The Action Space is conformed by all the possible words we can try on each step
        self.action_space = Discrete(len(WordleEnv.word_list))

        # Observation Space
        self.observation_space = Tuple([
            MultiBinary(len(WordleEnv.word_list)),  # Possible Words to try:             0 -> discarded word, 1 -> possible word
            MultiBinary(25),  # Possible Letters on position 0:    0 -> discarded letter, 1 -> possible letter
            MultiBinary(25),  # Possible Letters on position 1:    0 -> discarded letter, 1 -> possible letter
            MultiBinary(25),  # Possible Letters on position 2:    0 -> discarded letter, 1 -> possible letter
            MultiBinary(25),  # Possible Letters on position 3:    0 -> discarded letter, 1 -> possible letter
            MultiBinary(25),  # Possible Letters on position 4:    0 -> discarded letter, 1 -> possible letter
           # Discrete(6),  # Turns left
            MultiBinary(6)
        ])

    def reset(self, seed: int = None):
        self.seed: int = random.randint(0, len(WordleEnv.word_list) - 1) if seed is None else seed
        assert 0 <= self.seed < len(WordleEnv.word_list)
        self.answer: [chr] = [x for x in WordleEnv.word_list[self.seed]]

        self.remaining_turns = 6
        self.turns: [[chr]] = [['', '', '', '', ''] for x in range(6)]
        self.turn_scores: [[int]] = [[0, 0, 0, 0, 0] for x in range(6)]

        self.possible_words: [int] = [x for x in range(len(WordleEnv.word_list))]
        self.possible_letters: [[int]] = [[x for x in range(len(list(WordleEnv.letter_dict.values())))] for y in range(6)]

        self.done = False

        return self.get_state()

    def get_state(self):
        observation = [
            np.zeros(len(WordleEnv.word_list)),
            np.zeros(len(list(WordleEnv.letter_dict.values()))),
            np.zeros(len(list(WordleEnv.letter_dict.values()))),
            np.zeros(len(list(WordleEnv.letter_dict.values()))),
            np.zeros(len(list(WordleEnv.letter_dict.values()))),
            np.zeros(len(list(WordleEnv.letter_dict.values()))),
            np.ones(6)
            # 0
        ]

        turn = 6 - self.remaining_turns

        for x in range(len(WordleEnv.word_list)):
            if x in self.possible_words:
                observation[0][x] = 1
        for x in range(5):
            for i in self.possible_letters[x]:
                observation[x + 1][i] = 1
        assert len(self.turns[turn]) == len(self.turn_scores[turn])

        observation[6][turn] = 0

        return observation

    def step(self, action: int):

        turn = 6 - self.remaining_turns

        self.turns[turn] = [c for c in WordleEnv.word_list[action]]
        self.possible_words, self.turn_scores[turn] = self._check_word(action, self.possible_words)

        observation = self.get_state()
        self.remaining_turns -= 1

        done = True if self.turns[turn] == self.answer or self.remaining_turns == 0 else False
        reward = len(WordleEnv.word_list) / len(self.possible_words) * self.remaining_turns
        reward = reward * 10 if self.turns[turn] == self.answer else reward

        info = []

        return observation, reward, done, info

    def render(self):

        print(f'Turn {6 - self.remaining_turns}')
        for i in range(6):
            print('|', end='')
            for c in self.turns[i]:
                aux = '   ' if c == '' else f' {c} '
                print(f'{aux}|', end='')
            print('')
            print('|', end='')
            for s in self.turn_scores[i]:
                aux = f' {s} ' if s != -1 else f'{s} '
                print(f'{aux}|', end='')
            print('\n=====================')

    def _check_word(self, action, possible_words):

        word = WordleEnv.word_list[action]

        position = 0
        letters = []
        scores = []
        for letter in word:
            score = -1
            score = 5 if letter in self.answer else score
            score = position if letter == self.answer[position] else score
            position += 1

            letters.append(letter)
            scores.append(score)

        assert len(scores) == len(letters) == 5
        for _ in range(5):
            possible_words = self._check_letter(letters[_], scores[_], possible_words)

        return possible_words, scores

    def _check_letter(self, letter, position, possible_words):

        ans = []
        for pw in possible_words:

            w = WordleEnv.word_list[pw]
            letter_aux = WordleEnv.letter_dict[letter]

            if position >= 0:
                # The letter is in the answer word
                if position != 5:
                    # The specific position of the letter is known
                    # -> Keep words that contain letter in given position
                    if w[position] == letter:
                        ans.append(pw)
                    # Remove all letters except selected one from position
                    self.possible_letters[position] = [letter_aux]
                else:
                    # The specific position of the letter is unknown
                    # -> Keep words that contain letter
                    if letter in w:
                        ans.append(pw)
            else:
                # The letter isn't in the answer word
                # -> Keep words that don't contain the letter
                if letter not in w:
                    ans.append(pw)
                # Remove letter from all positions
                for x in range(6):
                    if letter_aux in self.possible_letters[x]:
                        self.possible_letters[x].remove(letter_aux)

        return ans

    def try_word(self, word):
        if word in WordleEnv.word_list:
            action = WordleEnv.word_list.index(word)
            self.step(action)
        else:
            print('The word does not exist in current WordRLe five')


if __name__ == '__main__':
    import ray
    import ray.rllib.agents.ppo as ppo
    from ray.tune.logger import pretty_print
    from ray import tune

    tune.register_env("my_env", lambda config: WordleEnv())
    ray.init()
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    trainer = ppo.PPOTrainer(config=config, env='my_env')

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(1000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 100 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)