"""
DQN Agent for Car Racing - Proper Implementation
Uses Double DQN with Target Network for stable learning.
"""
import random
from collections import deque

import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential, clone_model
from keras.optimizers import Adam

from settings import BATCH_SIZE, REPLAY_MEMORY


class DQNAgent:
    """
    Deep Q-Network Agent with:
    - Experience Replay
    - Target Network (Double DQN)
    - Epsilon-greedy exploration
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05  # Minimum exploration
        self.epsilon_decay = 0.9995  # Decay per step
        self.learning_rate = 0.0005  # Learning rate
        self.batch_size = BATCH_SIZE
        self.train_start = 2000  # Start training after this many steps

        # Target network update frequency
        self.target_update_freq = 100  # Update target network every N training steps
        self.train_step_count = 0

        # Replay Memory
        self.memory = deque(maxlen=REPLAY_MEMORY)

        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()  # Initialize target with same weights

    def _build_model(self):
        """Build neural network for Q-value approximation."""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='huber',  # Huber loss is more stable than MSE
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def update_target_model(self):
        """Copy weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory.
        
        Args:
            state: Preprocessed state vector (numpy array)
            action: Action index taken
            reward: Reward received
            next_state: Preprocessed next state vector (numpy array)
            done: Whether episode ended
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Preprocessed state vector (1D numpy array)
            training: If True, use epsilon-greedy; if False, greedy only
            
        Returns:
            Action index
        """
        # Epsilon-greedy exploration
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)

        # Ensure state is 2D: (1, state_size)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train the model using experience replay (batch learning)."""
        if len(self.memory) < self.train_start:
            return 0.0  # Not enough samples

        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample minibatch
        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare batch arrays
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])

        # Current Q values
        current_q = self.model.predict(states, verbose=0)

        # Next Q values from TARGET network (Double DQN)
        next_q_target = self.target_model.predict(next_states, verbose=0)
        
        # Best actions from MAIN network (for Double DQN)
        next_q_main = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(next_q_main, axis=1)

        # Update Q values using Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                current_q[i][actions[i]] = rewards[i]
            else:
                # Double DQN: Use main network to select action, target network for value
                current_q[i][actions[i]] = rewards[i] + self.gamma * next_q_target[i][best_actions[i]]

        # Train the model
        history = self.model.fit(states, current_q, batch_size=self.batch_size, 
                                  epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.update_target_model()

        return history.history['loss'][0]

    def save(self, filepath):
        """Save model weights to file."""
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights from file."""
        self.model.load_weights(filepath)
        self.update_target_model()  # Sync target network
        print(f"Model loaded from {filepath}")

    def save_full_model(self, filepath):
        """Save complete model (architecture + weights)."""
        self.model.save(filepath)
        print(f"Full model saved to {filepath}")