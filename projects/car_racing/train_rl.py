import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from collections import deque
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Import environment constants
from env import CarRacingEnv, ACTION_MAP
from settings import MODEL_PATH, DATA_PATH

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0001): # Lower learning rate for fine-tuning
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.98    # Higher gamma for long-term planning
        self.epsilon = 0.3   # Start with less randomness since we have a pre-trained model
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = learning_rate
        
        # Load behavior model and transfer weights
        self.model = self._build_model_from_h5()
        self.target_model = self._build_model_from_h5()
        self.update_target_model()

    def _build_model_from_h5(self):
        """Loads behavioral model and adapts it for RL (DQN)."""
        if not os.path.exists(MODEL_PATH):
            print("No existing model found. Building fresh RL model.")
            return self._build_fresh_model()

        print(f"Transferring knowledge from {MODEL_PATH}...")
        try:
            # Load the original behavioral cloning model
            base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
            # Create a new model for RL using the same architectural 'trunk'
            # We skip the last layer (which was classification) and add a Q-value layer
            new_model = tf.keras.Sequential()
            new_model.add(tf.keras.Input(shape=(self.state_size,)))
            
            # Transfer all layers except the last output layer
            for layer in base_model.layers[:-1]:
                new_model.add(layer)
            
            # Add the new DQN output layer (18 Q-values for actions)
            new_model.add(tf.keras.layers.Dense(128, activation='relu')) # Extra expressive layer
            new_model.add(tf.keras.layers.Dense(int(self.action_size), activation='linear'))
            
            new_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
            return new_model
        except Exception as e:
            print(f"Failed weight transfer: {e}. Building fresh.")
            return self._build_fresh_model()

    def _build_fresh_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            # tf.keras.layers.Dense(256, activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(int(self.action_size), activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0][0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3][0] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])

        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        self.model.fit(states, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        self.model.save(name)

def main():
    env = CarRacingEnv(render_mode="human" if "--render" in os.sys.argv else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    episodes = 2000
    batch_size = 64
    update_target_freq = 10
    save_freq = 50

    print("Improving weights via Reinforcement Learning...")

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        if hasattr(env, 'training_info'):
            env.training_info['Episode'] = e
            env.training_info['Epsilon'] = agent.epsilon

        for time_step in range(2000):
            if "--render" in os.sys.argv:
                if hasattr(env, 'training_info'):
                    env.training_info['Ep Reward'] = total_reward
                env.render()
                
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Ep: {e+1:4}/{episodes} | Score: {total_reward:8.2f} | Eps: {agent.epsilon:.3f} | Progress: {info.get('total_progress',0):.1f}")
                break
                
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if e % update_target_freq == 0:
            agent.update_target_model()

        if e > 0 and e % save_freq == 0:
            if not os.path.exists("models"): os.makedirs("models")
            agent.save(f"models/improved_car_dqn_ep{e}.keras")

    env.close()

if __name__ == "__main__":
    main()
