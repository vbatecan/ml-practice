"""
SAC Agent for Car Racing
Soft Actor-Critic (SAC) implementation for continuous control.
"""
import copy
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Concatenate, Dense, Input
from keras.optimizers import Adam
from settings import BATCH_SIZE, REPLAY_MEMORY

# Disable eager execution for performance if needed, but TF2 default is fine.
# tf.config.run_functions_eagerly(False)


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent.
    
    Features:
    - Continuous action space
    - Entropy regularization (automatic alpha tuning)
    - Double Q-Learning (two critics) to reduce overestimation
    - Soft target updates
    """

    def __init__(self, state_size, action_size, action_space_bounds=None):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space_bounds = action_space_bounds if action_space_bounds else [-1, 1]

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # Soft update rate
        self.learning_rate = 3e-4
        self.batch_size = BATCH_SIZE
        self.memory_buffer = deque(maxlen=REPLAY_MEMORY)
        self.train_start = 1000

        # Automatic Entropy Tuning
        self.target_entropy = -float(self.action_size)
        self.log_alpha = tf.Variable(0.0, dtype=tf.float32)
        self.alpha_optimizer = Adam(learning_rate=self.learning_rate)

        # Actor Network (Policy)
        self.actor = self._build_actor()
        self.actor_optimizer = Adam(learning_rate=self.learning_rate)

        # Critic Networks (Q1, Q2)
        self.critic_1 = self._build_critic()
        self.critic_2 = self._build_critic()
        self.critic_1_optimizer = Adam(learning_rate=self.learning_rate)
        self.critic_2_optimizer = Adam(learning_rate=self.learning_rate)

        # Target Critic Networks
        self.target_critic_1 = self._build_critic()
        self.target_critic_2 = self._build_critic()
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def _build_actor(self):
        """Gaussian Policy Network: inputs state, outputs mean and log_std."""
        inputs = Input(shape=(self.state_size,))
        x = Dense(256, activation="relu")(inputs)
        x = Dense(256, activation="relu")(x)
        
        mean = Dense(self.action_size, activation="linear")(x)
        log_std = Dense(self.action_size, activation="linear")(x)
        
        # Clip log_std to maintain stability
        # We don't use activation clip, but will clip inside the call if needed
        # Or simple Dense is fine, we handle bounding in `sample_action` logic usually
        # But here we just return the raw values model.
        
        return Model(inputs=inputs, outputs=[mean, log_std])

    def _build_critic(self):
        """Action-Value Network: inputs (state, action), outputs Q-value."""
        state_input = Input(shape=(self.state_size,))
        action_input = Input(shape=(self.action_size,))
        
        x = Concatenate()([state_input, action_input])
        x = Dense(256, activation="relu")(x)
        x = Dense(256, activation="relu")(x)
        q_value = Dense(1, activation="linear")(x)
        
        return Model(inputs=[state_input, action_input], outputs=q_value)

    def sample_action(self, state, deterministic=False):
        """Sample action from the policy."""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        mean, log_std = self.actor(state)
        
        if deterministic:
            action = tf.tanh(mean)
        else:
            std = tf.exp(log_std)
            normal = tf.random.normal(tf.shape(mean))
            # Reparameterization trick
            z = mean + std * normal
            action = tf.tanh(z)
            
        return action.numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        """Store experience tuple."""
        self.memory_buffer.append((state, action, reward, next_state, done))

    def train(self):
        """Perform one step of training on a batch."""
        if len(self.memory_buffer) < self.train_start:
            return 0.0

        batch = random.sample(self.memory_buffer, self.batch_size)
        
        states = np.array([x[0] for x in batch], dtype=np.float32)
        actions = np.array([x[1] for x in batch], dtype=np.float32)
        rewards = np.array([x[2] for x in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.array([x[3] for x in batch], dtype=np.float32)
        dones = np.array([x[4] for x in batch], dtype=np.float32).reshape(-1, 1)

        loss_info = self._train_step(states, actions, rewards, next_states, dones)
        return loss_info

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """TF Function for optimized training step."""
        
        # 1. Update Critic
        with tf.GradientTape(persistent=True) as tape:
            # Target actions (next state)
            next_means, next_log_stds = self.actor(next_states)
            next_stds = tf.exp(next_log_stds)
            next_noise = tf.random.normal(tf.shape(next_means))
            next_zs = next_means + next_stds * next_noise
            next_actions = tf.tanh(next_zs)
            
            # Compute likelihood for entropy term (in target)
            # log_prob = -0.5 * (noise^2 + log(2pi) + 2*log_std) - log(1 - tanh(z)^2)
            # Simplified:
            log_probs = self._compute_log_probs(next_means, next_log_stds, next_zs)
            
            # Target Q-values
            q1_target = self.target_critic_1([next_states, next_actions])
            q2_target = self.target_critic_2([next_states, next_actions])
            min_q_target = tf.minimum(q1_target, q2_target)
            
            # Soft Q-target
            soft_q_target = min_q_target - self.alpha * log_probs
            y = rewards + self.gamma * (1.0 - dones) * soft_q_target
            
            # Current Q-values
            q1_pred = self.critic_1([states, actions])
            q2_pred = self.critic_2([states, actions])
            
            critic_1_loss = tf.reduce_mean(tf.square(y - q1_pred))
            critic_2_loss = tf.reduce_mean(tf.square(y - q2_pred))
            critic_loss = critic_1_loss + critic_2_loss

        # Apply Critic Gradients
        c1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        c2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(c1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(c2_grads, self.critic_2.trainable_variables))
        
        # 2. Update Actor
        with tf.GradientTape() as tape:
            means, log_stds = self.actor(states)
            stds = tf.exp(log_stds)
            noise = tf.random.normal(tf.shape(means))
            zs = means + stds * noise
            sampled_actions = tf.tanh(zs)
            
            log_probs = self._compute_log_probs(means, log_stds, zs)
            
            # Q-values for sampled actions
            q1_pi = self.critic_1([states, sampled_actions])
            q2_pi = self.critic_2([states, sampled_actions])
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            
            actor_loss = tf.reduce_mean(self.alpha * log_probs - min_q_pi)

        # Apply Actor Gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # 3. Update Alpha (Entropy Temperature)
        with tf.GradientTape() as tape:
            # We want alpha * (-log_prob - target_entropy)
            # Recompute log_probs to detach graph if needed, but TF handles it.
            # Using the log_probs from current policy step
            alpha_loss = tf.reduce_mean(-self.alpha * (log_probs + self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        # 4. Soft Update Targets
        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)
        
        return critic_loss + actor_loss

    def _compute_log_probs(self, means, log_stds, zs):
        """
        Compute log probabilities for the squashed Gaussian distribution.
        
        log_prob = log_prob_gaussian - sum(log(1 - tanh(z)^2))
        """
        stds = tf.exp(log_stds)
        # Normal distribution log_prob
        # z = (zs - means) / stds  (which is just the noise used)
        # But we need density of zs under N(means, stds)
        # log_prob = -0.5 * ((x - mu)/sigma)^2 - log(sigma) - 0.5 * log(2pi)
        
        # We can implement manually:
        noise = (zs - means) / (stds + 1e-6)
        log_prob_n = -0.5 * tf.square(noise) - log_stds - 0.5 * np.log(2 * np.pi)
        log_prob_n = tf.reduce_sum(log_prob_n, axis=1, keepdims=True)
        
        # Correction for Tanh squashing
        # log(1 - tanh(z)^2)
        # Numerically stable: 2 * (log(2) - z - softplus(-2z))
        tanh_z = tf.tanh(zs)
        log_correction = tf.reduce_sum(tf.math.log(1.0 - tf.square(tanh_z) + 1e-6), axis=1, keepdims=True)
        
        return log_prob_n - log_correction

    def _soft_update(self, source, target):
        """Soft update model weights: target = tau * source + (1 - tau) * target"""
        for s_w, t_w in zip(source.trainable_variables, target.trainable_variables):
            t_w.assign(self.tau * s_w + (1.0 - self.tau) * t_w)

    def save_models(self, base_path_prefix):
        """Save all models for training resume."""
        self.actor.save_weights(f"{base_path_prefix}_actor.weights.h5")
        self.critic_1.save_weights(f"{base_path_prefix}_critic1.weights.h5")
        self.critic_2.save_weights(f"{base_path_prefix}_critic2.weights.h5")
        self.target_critic_1.save_weights(f"{base_path_prefix}_target1.weights.h5")
        self.target_critic_2.save_weights(f"{base_path_prefix}_target2.weights.h5")
        # Save log_alpha as numpy
        np.save(f"{base_path_prefix}_log_alpha.npy", self.log_alpha.numpy())

    def load_models(self, base_path_prefix):
        """Load all models for training resume."""
        try:
            self.actor.load_weights(f"{base_path_prefix}_actor.weights.h5")
            self.critic_1.load_weights(f"{base_path_prefix}_critic1.weights.h5")
            self.critic_2.load_weights(f"{base_path_prefix}_critic2.weights.h5")
            self.target_critic_1.load_weights(f"{base_path_prefix}_target1.weights.h5")
            self.target_critic_2.load_weights(f"{base_path_prefix}_target2.weights.h5")
            
            try:
                alpha_val = np.load(f"{base_path_prefix}_log_alpha.npy")
                self.log_alpha.assign(float(alpha_val))
            except:
                print("Warning: Could not load log_alpha, starting entropy from scratch.")
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def save_full_model(self, filepath):
        """Save inference-only model (Actor Mean) for game usage."""
        # Create a new model that outputs only the mean (action)
        # This ensures main.py receives a simple (batch, action) array
        # and doesn't break due to list output or multiple return values.
        
        # We share the layers from the trained actor
        inputs = self.actor.input
        # The actor outputs [mean, log_std]. We want index 0.
        mean_output = self.actor.outputs[0]
        
        inference_model = Model(inputs=inputs, outputs=mean_output)
        inference_model.save(filepath)
    
    def save_memory(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            # Save as list to avoid pickle issues with deque sometimes
            pickle.dump(list(self.memory_buffer), f)

    def load_memory(self, filepath):
        import os
        import pickle
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.memory_buffer = deque(pickle.load(f), maxlen=REPLAY_MEMORY)
            return True
        return False