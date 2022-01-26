# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Austral Intelligence Labs 01. WordRLe - Training the Model
------------------------------------------------------------------------------------------------------------------------
Date: 21 of January, 2022
Considerations:
    - Tensorflow Keras model to train RL NN to play Wordle

Authors:
        Ignacio Brottier González           ignacio.brottier@accenture.com
"""

# ----------------------------------------------------------------------------------------------------------------------
# Importing libraries and necessary functions
# ----------------------------------------------------------------------------------------------------------------------

from main import WordleEnv
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from collections import deque

class DQN:
    def __init__(self, env):
        self.env     = env
        self.memory  = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        #self.epsilon_decay = 0.995
        self.epsilon_decay = 0.7
        self.learning_rate = 0.005
        self.tau = .125

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape  = sum([self.env.observation_space.spaces[x].shape[0] for x in range(7)]) #+ 1
        model.add(Dense(24, input_dim=state_shape, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        #np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

if __name__ == '__main__':
    env     = WordleEnv()
    gamma   = 0.9
    epsilon = .95

    trials  = 1000
    trial_len = 6

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        env.reset()
        cur_state = env.get_state()
        print(f'TRIAL: {trial + 1}')
        print(f'\tWORD: {env.answer}')
        env.render()
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            new_state = new_state
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            env.render()
            if done:
                if env.turns[step] == env.answer:
                    print(f'Correct answer found in {step} turns')
                else:
                    print(f'Correct answer not found. Possible words remaining:')
                    print([WordleEnv.word_list[x] for x in range(len(cur_state[0])) if cur_state[0][x] == 1])
                break
        if env.turns[step] == env.answer:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break