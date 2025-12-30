"""
Reinforcement Learning Training Script for Car Racing
Implements proper DQN training loop with:
- State preprocessing
- Reward shaping
- Episode logging
- Model checkpointing
- Training resume support
- Graceful shutdown with memory save
"""

import json
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np

from agent import DQNAgent
from main import Game
from settings import RADAR_COUNT, RL_HEADLESS

# Checkpoint paths
CHECKPOINT_DIR = "save/rl_models"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.json")
LATEST_MODEL_FILE = os.path.join(CHECKPOINT_DIR, "latest_model.keras")
MEMORY_FILE = os.path.join(CHECKPOINT_DIR, "replay_memory.pkl")

# Global reference for signal handler
_current_agent = None
_current_episode = 0
_current_epsilon = 1.0
_best_reward = float('-inf')
_total_steps = 0


def graceful_shutdown(signum, frame):
    """Handle Ctrl+C gracefully by saving state before exit."""
    global _current_agent, _current_episode, _current_epsilon, _best_reward, _total_steps
    print("\n\n⚠️  Interrupt received! Saving state...")
    
    if _current_agent is not None:
        # Save model
        _current_agent.save_full_model(LATEST_MODEL_FILE)
        # Save memory
        _current_agent.save_memory(MEMORY_FILE)
        # Save checkpoint
        save_checkpoint(_current_episode, _current_epsilon, _best_reward, _total_steps)
        print("✓ State saved successfully. Safe to exit.")
    
    sys.exit(0)


def save_checkpoint(episode, epsilon, best_reward, total_steps):
    """Save training checkpoint to JSON file."""
    checkpoint = {
        "episode": episode,
        "epsilon": epsilon,
        "best_reward": best_reward,
        "total_steps": total_steps,
    }
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint():
    """Load training checkpoint from JSON file.

    Returns:
        dict with checkpoint data or None if no checkpoint exists
    """
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return None
    return None


def preprocess_state(state_dict):
    """
    Convert game state dictionary to normalized feature vector.

    Features:
        - speed (normalized by max speed)
        - angle (normalized to [-1, 1])
        - acceleration (normalized)
        - cte (cross-track error, normalized)
        - heading_error (normalized to [-1, 1])
        - radars (log-scaled and normalized)

    Returns:
        numpy array of shape (state_size,)
    """
    # 1. Log-scale and normalize radars
    radars = np.array(state_dict["radars"], dtype=np.float32)
    radars_log = np.log1p(radars) / 7.0  # log(1000) ≈ 7

    # 2. Normalize continuous features
    speed = state_dict["speed"] / 2000.0  # MAX_SPEED
    angle = state_dict["angle"] / 360.0  # Normalize to [-1, 1]
    acceleration = np.clip(state_dict["acceleration"] / 500.0, -1, 1)  # ACCELERATION
    cte = np.clip(state_dict["cte"] / 100.0, -1, 1)  # Normalize CTE
    heading = np.clip(state_dict["heading_error"] / 180.0, -1, 1)  # Degrees

    # 3. Combine into feature vector
    # Order: [speed, angle, acceleration, cte, heading, R0...Rn]
    base_features = np.array(
        [speed, angle, acceleration, cte, heading], dtype=np.float32
    )

    return np.concatenate([base_features, radars_log])


# Track slow time for progressive penalty
_slow_steps_counter = 0
_SLOW_THRESHOLD = 15  # Speed below this is "slow"
_GRACE_PERIOD_STEPS = 180  # 3 seconds at 60 FPS before penalty starts


def reset_slow_counter():
    """Reset slow counter at episode start."""
    global _slow_steps_counter
    _slow_steps_counter = 0


def calculate_reward(state_dict, prev_state_dict, action_idx):
    """
    Calculate shaped reward based on driving behavior.

    Reward components:
        + Progress on track (speed-based)
        + Staying centered on track
        + Aligning with track direction
        - Collisions (heavy penalty)
        - Driving too slow for too long (progressive time-based penalty)
        - Excessive steering while moving fast

    Returns:
        float: reward value
    """
    global _slow_steps_counter
    reward = 0.0

    # === TERMINAL CONDITION: COLLISION ===
    if state_dict["collided"]:
        return -100.0  # Heavy penalty

    # === PROGRESS REWARD ===
    speed = state_dict["speed"]
    
    # Track time spent slow
    if speed < _SLOW_THRESHOLD:
        _slow_steps_counter += 1
    else:
        _slow_steps_counter = 0  # Reset when moving at good speed
    
    # Reward for good speed
    if speed > 50:
        reward += speed * 0.005  # Moderate speed bonus
    
    # === TIME-BASED SLOW PENALTY ===
    if speed < _SLOW_THRESHOLD:
        if _slow_steps_counter > _GRACE_PERIOD_STEPS:
            # After 3 seconds: progressive penalty
            overtime_seconds = (_slow_steps_counter - _GRACE_PERIOD_STEPS) / 60.0
            # Progressive penalty: starts at -0.5, grows to -3.0 over 5 seconds
            slow_penalty = min(0.1 + overtime_seconds * 0.1, 2.0)
            reward -= slow_penalty
        else:
            # During grace period: still a small penalty (not zero!)
            reward -= 0.21

    # === TRACK ALIGNMENT ===
    # Penalize being far from center (cross-track error)
    cte = abs(state_dict["cte"])
    if cte < 25 and cte > -25:
        reward += 0.1  # Bonus for staying centered
    elif cte > 30 or cte < -30:
        reward -= cte * 0.01  # Progressive penalty

    # Small penalty for aggressive steering at high speed
    if speed > 125 and action_idx in [1, 2]:  # Left or Right with forward
        reward -= 0.1

    # Reward maintaining speed
    if prev_state_dict is not None:
        prev_speed = prev_state_dict["speed"]
        speed_diff = abs(speed - prev_speed)
        if speed_diff < 35:
            reward += 0.1

    # === WALL PROXIMITY PENALTY ===
    # Penalize being too close to walls using sensor readings
    radars = state_dict["radars"]
    danger_threshold = 50 # Very close to wall
    warning_threshold = 100  # Getting close
    
    for radar_dist in radars:
        if radar_dist < danger_threshold:
            # Heavy penalty for being very close
            reward -= 0.5
        elif radar_dist < warning_threshold:
            # Progressive penalty as distance decreases
            proximity_penalty = (warning_threshold - radar_dist) / warning_threshold
            reward -= proximity_penalty * 0.3

    return reward


def map_action_to_controls(action_idx):
    """
    Map discrete action index to control tuple.

    Action Space (5 discrete actions):
        0: Forward (full throttle)
        1: Forward + Left
        2: Forward + Right
        3: Brake/Reverse
        4: Coast (no input)

    Returns:
        list: [w, a, s, d] control values
    """
    mapping = {
        0: [1, 0, 0, 0],  # Forward
        1: [1, 1, 0, 0],  # Forward + Left
        2: [1, 0, 0, 1],  # Forward + Right
        3: [0, 0, 1, 0],  # Brake
        4: [0, 0, 0, 0],  # Coast
    }
    return mapping.get(action_idx, [0, 0, 0, 0])


class TrainingMetrics:
    """Tracks and displays live training metrics."""

    def __init__(self):
        self.episode = 0
        self.step = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.epsilon = 1.0
        self.loss = 0.0
        self.speed = 0.0
        self.action = 0
        self.memory_size = 0
        self.avg_reward_100 = 0.0

    def to_dict(self):
        """Return metrics as dictionary for UI display."""
        return {
            "episode": self.episode,
            "step": self.step,
            "total_steps": self.total_steps,
            "episode_reward": self.episode_reward,
            "epsilon": self.epsilon,
            "loss": self.loss,
            "speed": self.speed,
            "action": self.action,
            "memory_size": self.memory_size,
            "avg_reward_100": self.avg_reward_100,
        }


class TrainingLogger:
    """Simple logger for tracking training progress."""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.csv")

        # Write header
        with open(self.log_file, "w") as f:
            f.write("episode,steps,reward,avg_reward_100,epsilon,loss,duration\n")

        self.rewards_history = []

    def log_episode(self, episode, steps, reward, epsilon, loss, duration):
        """Log episode statistics."""
        self.rewards_history.append(reward)

        # Calculate running average
        avg_reward = np.mean(self.rewards_history[-100:])

        with open(self.log_file, "a") as f:
            f.write(
                f"{episode},{steps},{reward:.2f},{avg_reward:.2f},{epsilon:.4f},{loss:.6f},{duration:.2f}\n"
            )

        return avg_reward


def train(resume=False):
    """Main training loop.

    Args:
        resume: If True, attempt to resume from last checkpoint
    """
    global _current_agent, _current_episode, _current_epsilon, _best_reward, _total_steps
    
    # Register graceful shutdown handler
    signal.signal(signal.SIGINT, graceful_shutdown)
    
    print("=" * 60)
    print("DQN Training for Car Racing")
    print("=" * 60)

    # Initialize game and agent
    game = Game()

    # Get initial state to determine state size
    initial_state = game.reset_game()
    state_size = 5 + RADAR_COUNT  # 5 base features + radars
    action_size = 5  # 5 discrete actions

    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Headless mode: {RL_HEADLESS}")

    agent = DQNAgent(state_size, action_size)
    _current_agent = agent  # Set global reference for signal handler
    
    logger = TrainingLogger()

    # Training parameters
    num_episodes = 500
    max_steps_per_episode = 2000
    save_interval = 50  # Save model every N episodes

    # Create save directory
    save_dir = CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)

    best_reward = float("-inf")
    total_steps = 0
    start_episode = 1

    # Try to resume from checkpoint
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint and os.path.exists(LATEST_MODEL_FILE):
            start_episode = checkpoint["episode"] + 1
            agent.epsilon = checkpoint["epsilon"]
            best_reward = checkpoint["best_reward"]
            total_steps = checkpoint["total_steps"]

            # Load the model weights
            try:
                from keras.models import load_model

                agent.model = load_model(LATEST_MODEL_FILE)
                agent.update_target_model()
                
                # Try to load replay memory
                memory_loaded = agent.load_memory(MEMORY_FILE)
                
                # Only boost epsilon if memory is empty (needs warmup)
                if not memory_loaded or len(agent.memory) < agent.train_start:
                    original_epsilon = agent.epsilon
                    agent.epsilon = max(agent.epsilon, 1.0)
                    print(f"  Epsilon: {original_epsilon:.4f} → {agent.epsilon:.4f} (boosted, memory needs refill)")
                else:
                    print(f"  Epsilon: {agent.epsilon:.4f} (memory restored, no boost needed)")
                
                print(f"\n✓ Resumed from episode {start_episode - 1}")
                print(f"  Best reward: {best_reward:.2f}")
                print(f"  Total steps: {total_steps}")
                print(f"  Memory size: {len(agent.memory)}")
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
                print("Starting fresh training...")
                start_episode = 1
                agent.epsilon = 1.0
                best_reward = float("-inf")
                total_steps = 0
        else:
            print("No checkpoint found. Starting fresh training...")

    # Update global state for signal handler
    _best_reward = best_reward
    _total_steps = total_steps

    print(f"\nStarting training from episode {start_episode} to {num_episodes}...")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("Press Ctrl+C to save and exit gracefully")
    print("-" * 60)

    # Live metrics tracking
    metrics = TrainingMetrics()
    log_interval = 100  # Print to terminal every N steps

    for episode in range(start_episode, num_episodes + 1):
        # Reset environment
        state_dict = game.reset_game()
        reset_slow_counter()  # Reset time-based slow penalty tracker
        state = preprocess_state(state_dict)

        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        prev_state_dict = None

        start_time = time.time()
        metrics.episode = episode

        for step in range(max_steps_per_episode):
            # Select action
            action = agent.act(state, training=True)
            controls = map_action_to_controls(action)

            # Take action in environment
            next_state_dict, _, done, info = game.step(controls, metrics.to_dict())
            next_state = preprocess_state(next_state_dict)

            # Calculate shaped reward
            reward = calculate_reward(next_state_dict, prev_state_dict, action)

            # Store experience (preprocessed states!)
            agent.remember(state, action, reward, next_state, done)

            # Train (experience replay) - only every N steps for performance
            # This reduces GPU overhead while maintaining learning quality
            if total_steps % agent.train_frequency == 0:
                loss = agent.replay()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1

            # Update state
            state = next_state
            prev_state_dict = state_dict
            state_dict = next_state_dict
            episode_reward += reward
            total_steps += 1
            
            # Update global state for signal handler
            _total_steps = total_steps
            _current_epsilon = agent.epsilon

            # Update live metrics
            metrics.step = step + 1
            metrics.total_steps = total_steps
            metrics.episode_reward = episode_reward
            metrics.epsilon = agent.epsilon
            metrics.loss = episode_loss / max(loss_count, 1)
            metrics.speed = next_state_dict["speed"]
            metrics.action = int(action)
            metrics.memory_size = len(agent.memory)
            metrics.avg_reward_100 = (
                np.mean(logger.rewards_history[-100:]) if logger.rewards_history else 0
            )

            # Periodic terminal logging
            if total_steps % log_interval == 0:
                print(
                    f"  [Step {total_steps:7d}] "
                    f"Ep {episode} | "
                    f"Step {step + 1:4d} | "
                    f"Reward: {episode_reward:7.1f} | "
                    f"Speed: {metrics.speed:5.0f} | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"Mem: {len(agent.memory):5d}"
                )

            if done:
                break

        # Episode complete
        duration = time.time() - start_time
        avg_loss = episode_loss / max(loss_count, 1)

        # Log episode
        avg_reward = logger.log_episode(
            episode, step + 1, episode_reward, agent.epsilon, avg_loss, duration
        )

        # Print progress
        print(
            f"Episode {episode:4d} | "
            f"Steps: {step + 1:5d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Avg(100): {avg_reward:8.2f} | "
            f"ε: {agent.epsilon:.4f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {duration:.1f}s"
        )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_full_model(os.path.join(save_dir, "best_model.keras"))

        # Update global state for signal handler
        _current_episode = episode
        _best_reward = best_reward
        _current_epsilon = agent.epsilon

        # Save checkpoint after EVERY episode for safe resume
        agent.save_full_model(LATEST_MODEL_FILE)
        save_checkpoint(episode, agent.epsilon, best_reward, total_steps)
        
        # Save memory every 10 episodes (not every episode - it's slow)
        if episode % 10 == 0:
            agent.save_memory(MEMORY_FILE)

        # Periodic save with episode number (milestone)
        if episode % save_interval == 0:
            agent.save_full_model(os.path.join(save_dir, f"model_ep{episode}.keras"))
            print(f"  → Milestone checkpoint saved at episode {episode}")

    # Final save
    agent.save_full_model(os.path.join(save_dir, "final_model.keras"))
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total steps: {total_steps}")
    print(f"Best episode reward: {best_reward:.2f}")
    print(f"Models saved to: {save_dir}")
    print("=" * 60)


def evaluate(model_path, num_episodes=10):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        num_episodes: Number of evaluation episodes
    """
    print("=" * 60)
    print("Evaluating Trained Model")
    print("=" * 60)

    game = Game()

    state_size = 5 + RADAR_COUNT
    action_size = 5

    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation

    total_rewards = []
    total_steps = []

    for episode in range(1, num_episodes + 1):
        state_dict = game.reset_game()
        state = preprocess_state(state_dict)

        episode_reward = 0.0
        prev_state_dict = None

        for step in range(3000):
            action = agent.act(state, training=False)
            controls = map_action_to_controls(action)

            next_state_dict, _, done, info = game.step(controls)
            next_state = preprocess_state(next_state_dict)

            reward = calculate_reward(next_state_dict, prev_state_dict, action)

            state = next_state
            prev_state_dict = state_dict
            state_dict = next_state_dict
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)
        total_steps.append(step + 1)

        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Steps = {step + 1}")

    print("-" * 60)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(total_steps):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode: python train_rl.py eval [model_path] [num_episodes]
        model_path = (
            sys.argv[2] if len(sys.argv) > 2 else "save/rl_models/best_model.keras"
        )
        num_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        evaluate(model_path, num_eps)
    elif len(sys.argv) > 1 and sys.argv[1] == "resume":
        # Resume mode: python train_rl.py resume
        train(resume=True)
    else:
        # Fresh training mode: python train_rl.py
        train(resume=False)
