import random
from collections import deque

import numpy as np
from keras.layers import Dense, Input
from keras.src.models import Sequential
from keras.src.optimizers import Adam

from settings import *


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.epsilon = 0.998
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.memory = deque(maxlen=REPLAY_MEMORY)
        self.batch_size: int = BATCH_SIZE
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(
            [
                Input((self.state_size,)),
                Dense(64, activation="relu"),
                Dense(64, activation="relu"),
                Dense(self.action_size, activation="linear"),
            ]
        )
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        act_values = self.model.predict(state)[0]
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Bellman Equation: r + gamma * max(Q(s', a'))
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
            
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay