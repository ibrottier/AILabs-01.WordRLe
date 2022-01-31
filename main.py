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

class WordleEnv(Env):
    word_list = [row['HEADER'] for _, row in pd.read_csv(r'C:\Users\patricio.ivan.pipp\Documents\RL\AILabs-01.WordRLe\five_letter_words.txt').iterrows()]
    letter_dict = {row['IDX']: row['VALUE'] for _, row in pd.read_csv(r'C:\Users\patricio.ivan.pipp\Documents\RL\AILabs-01.WordRLe\letters.txt').iterrows()}

    def __init__(self, seed: int = None):

        # Initial State
        self.seed: int = random.randint(0, len(WordleEnv.word_list) - 1) if seed is None else seed
        assert 0 <= self.seed < len(WordleEnv.word_list)
        self.answer: [int] = [WordleEnv.letter_dict[x] for x in WordleEnv.word_list[self.seed]]
        self.remaining_turns = 6
        self.possible_words: [int] = [x for x in range(len(WordleEnv.word_list))]
        self.possible_letters: [[int]] = [[] for y in range(5)]

        self.done = False
        self.found_word = False

        self.scores: [[int]] = [[0 for x in range(5)] for y in range(6)]
        self.turns: [[chr]] = [['' for x in range(5)] for y in range(6)]
        # Action Space
        # The Action Space is conformed by 5 position vector with 26 (in english) possible values for each position
        self.action_space = MultiDiscrete([len(list(WordleEnv.letter_dict.values())) for x in range(5)])

        # Observation Space
        # The state/observation is composed by the possible letters for each position
        self.state = self._set_initial_state()
        self.observation_space = Tuple([
            MultiBinary(len(list(WordleEnv.letter_dict.values()))),
            MultiBinary(len(list(WordleEnv.letter_dict.values()))),
            MultiBinary(len(list(WordleEnv.letter_dict.values()))),
            MultiBinary(len(list(WordleEnv.letter_dict.values()))),
            MultiBinary(len(list(WordleEnv.letter_dict.values())))
        ])

    def reset(self, seed: int = None):
        self.seed: int = random.randint(0, len(WordleEnv.word_list) - 1) if seed is None else seed
        assert 0 <= self.seed < len(WordleEnv.word_list)
        self.answer: [int] = [WordleEnv.letter_dict[x] for x in WordleEnv.word_list[self.seed]]
        self.remaining_turns = 6
        self.possible_words: [int] = [x for x in range(len(WordleEnv.word_list))]
        self.possible_letters: [[int]] = [[] for y in range(5)]

        self.done = False
        self.found_word = False

        self.scores: [[int]] = [[0 for x in range(5)] for y in range(6)]
        self.turns: [[chr]] = [['' for x in range(5)] for y in range(6)]

        self.state = self._set_initial_state()

        return self.state

    def _set_initial_state(self):

        self.state = [np.zeros(len(list(WordleEnv.letter_dict.values()))) for x in range(5)]
        initial_state = []
        for _ in range(5):
            ans = []
            for word in WordleEnv.word_list:
                if word[_] not in ans:
                    aux = WordleEnv.letter_dict[word[_]]
                    self.possible_letters[_].append(aux)
                    ans.append(aux)
                if len(ans) == len(list(WordleEnv.letter_dict.keys())):
                    break
            initial_state.append(ans)

        self.state = [np.zeros(len(list(WordleEnv.letter_dict.values()))) for x in range(5)]
        for _ in range(5):
            for l in initial_state[_]:
                self.state[_][l] = 1

        return self.state

    def get_reward(self, option):

        turn = 6 - self.remaining_turns

        if option == 0:
            reward = len(WordleEnv.word_list) / len(self.possible_words) * self.remaining_turns
            reward = reward * 10 if self.found_word else reward
        elif option == 1:
            reward = 1 / len(self.possible_words) * self.remaining_turns
            reward = reward * 10 if self.found_word else reward
        elif option == 2:
            reward = sum([(self.scores[turn][x] - 1) * 5 for x in range(5)]) * self.remaining_turns
            reward = reward * 10 if self.found_word else reward
        else:
            raise ValueError(f'Option {option} not mapped in get_reward method')

        return reward

    def _get_action_validity(self, action: [int]):
        word = [list(WordleEnv.letter_dict.keys())[x] for x in action]
        word = ''.join(word)
        if word in WordleEnv.word_list:
            return True, word
        else:
            return False, word

    def _get_valid_action(self):

        word = WordleEnv.word_list[random.randint(0, len(WordleEnv.word_list)-1)]
        action = [WordleEnv.letter_dict[x] for x in word]

        return action


    def step(self, action: [int]):

        validity, word = self._get_action_validity(action)
        if not validity:
            action = self._get_valid_action()
            reward_aux = -100
        else:
            reward_aux = 0

        turn = 6 - self.remaining_turns

        self.turns[turn] = action
        self.possible_words, self.scores[turn] = self._check_word(action, self.possible_words)

        self.found_word = True if all([int(self.turns[turn][x]) == self.answer[x] for x in range(5)]) else False
        reward = self.get_reward(2) if validity else -100
        # reward += reward_aux

        self.state = [np.zeros(len(list(WordleEnv.letter_dict.values()))) for x in range(5)]
        for _ in range(5):
            for l in self.possible_letters[_]:
                self.state[_][l] = 1

        self.remaining_turns -= 1
        self.done = True if self.found_word else False
        self.done = True if self.remaining_turns == 0 else self.done


        info = {
            'answer': self.get_answer(),
            'action_id': action,
            'action_word': word,
            'reward': reward,
            'found_word': self.found_word,
            'remaining_turns': self.remaining_turns
        }

        return self.state, reward, self.done, info

    def render(self):

        turn = 6 - self.remaining_turns
        print(f'\t\tTurn {turn}')
        for i in range(turn):
            print(self.get_word(self.turns[i]))
            print(self.scores[i] )


    def _check_word(self, action, possible_words):

        valid, word = self._get_action_validity(action)

        if valid:
            position = 0
            letters = []
            scores = []
            for letter in word:
                score = 1
                score = 2 if WordleEnv.letter_dict[letter] in self.answer else score
                score = 3 if WordleEnv.letter_dict[letter] == self.answer[position] else score
                position += 1

                letters.append(letter)
                scores.append(score)

            assert len(scores) == len(letters) == 5
            for _ in range(5):
                possible_words = self._check_letter(letters[_], scores[_], _, possible_words)

            return possible_words, scores
        else:
            raise ValueError(f'Word {word} not valid')

    def _check_letter(self, letter, score, position, possible_words):

        ans = []
        assert 0 < score <= 3
        assert 0 <= position <= 4
        assert letter in WordleEnv.letter_dict.keys()

        for pw in possible_words:

            w = WordleEnv.word_list[pw]
            letter_aux = WordleEnv.letter_dict[letter]

            # The letter is in the answer word
            if score == 3:
                # The specific position of the letter is known
                # -> Keep words that contain letter in given position
                if w[position] == letter:
                    ans.append(pw)
                # Remove all letters except selected one from position
                self.possible_letters[position] = [letter_aux]
            elif score == 2:
                # The specific position of the letter is unknown
                # -> Keep words that contain letter
                if letter in w:
                    ans.append(pw)

            elif score == 1:
                # The letter isn't in the answer word
                # -> Keep words that don't contain the letter
                if letter not in w:
                    ans.append(pw)
                # Remove letter from all positions
                for x in range(5):
                    if letter_aux in self.possible_letters[x]:
                        self.possible_letters[x].remove(letter_aux)

        return ans

    def get_word(self, word:[int]):
        ans = [list(WordleEnv.letter_dict.keys())[c] for c in word]
        return ans

    def get_answer(self):
        return self.get_word(self.answer)

    def try_word(self, word):
        if word in WordleEnv.word_list:
            action = [ WordleEnv.letter_dict[c] for c in word]
            self.step(action)
        else:
            print('The word does not exist in current WordRLe five')


if __name__ == '__main__':
    import ray
    import ray.rllib.agents.ppo as ppo
    from ray.tune.logger import pretty_print
    from ray import tune
    import logging

    tune.register_env("my_env", lambda config: WordleEnv())
    ray.init(logging_level=logging.DEBUG)
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 4
    trainer = ppo.PPOTrainer(config=config, env='my_env')

    # Can optionally call trainer.restore(path) to load a checkpoint.
    trainer.restore(r'C:\Users\patricio.ivan.pipp\ray_results\PPO_my_env_2022-01-28_17-53-32w2cu9v5s\checkpoint_000593\checkpoint-593')

    for i in range(2000):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()
        print(pretty_print(result))

        if i % 10 == 0:
            checkpoint = trainer.save()
            print("checkpoint saved at", checkpoint)