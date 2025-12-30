"""
Reinforcement Learning Training Script for Car Racing
Implements proper DQN training loop with:
- State preprocessing
- Reward shaping
- Episode logging
- Model checkpointing
"""
import json
import os
import time
from datetime import datetime

import numpy as np

from agent import DQNAgent
from main import Game
from settings import RADAR_COUNT, RL_HEADLESS


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
    base_features = np.array([speed, angle, acceleration, cte, heading], dtype=np.float32)
    
    return np.concatenate([base_features, radars_log])


def calculate_reward(state_dict, prev_state_dict, action_idx):
    """
    Calculate shaped reward based on driving behavior.
    
    Reward components:
        + Progress on track (speed-based)
        + Staying centered on track
        + Aligning with track direction
        - Collisions (heavy penalty)
        - Driving too slow (small penalty)
        - Excessive steering while moving fast
        
    Returns:
        float: reward value
    """
    reward = 0.0
    
    # === TERMINAL CONDITION: COLLISION ===
    if state_dict["collided"]:
        return -100.0  # Heavy penalty
    
    # === PROGRESS REWARD ===
    # Encourage forward movement
    speed = state_dict["speed"]
    if speed > 50:
        reward += speed * 0.01  # Scale down
    elif speed < 30:
        reward -= 0.5  # Penalty for being too slow
    
    # === TRACK ALIGNMENT ===
    # Penalize being far from center (cross-track error)
    cte = abs(state_dict["cte"])
    if cte < 30:
        reward += 0.5  # Bonus for staying centered
    elif cte > 80:
        reward -= cte * 0.01  # Progressive penalty
    
    # Penalize wrong heading
    heading_error = abs(state_dict["heading_error"])
    if heading_error < 15:
        reward += 0.3  # Bonus for good alignment
    elif heading_error > 45:
        reward -= heading_error * 0.005
    
    # === ACTION SMOOTHNESS ===
    # Small penalty for aggressive steering at high speed
    if speed > 500 and action_idx in [1, 2]:  # Left or Right with forward
        reward -= 0.1
    
    # === SPEED CONSISTENCY ===
    # Reward maintaining speed
    if prev_state_dict is not None:
        prev_speed = prev_state_dict["speed"]
        speed_diff = abs(speed - prev_speed)
        if speed_diff < 20:
            reward += 0.1
    
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
            'episode': self.episode,
            'step': self.step,
            'total_steps': self.total_steps,
            'episode_reward': self.episode_reward,
            'epsilon': self.epsilon,
            'loss': self.loss,
            'speed': self.speed,
            'action': self.action,
            'memory_size': self.memory_size,
            'avg_reward_100': self.avg_reward_100,
        }


class TrainingLogger:
    """Simple logger for tracking training progress."""
    
    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.csv")
        
        # Write header
        with open(self.log_file, 'w') as f:
            f.write("episode,steps,reward,avg_reward_100,epsilon,loss,duration\n")
        
        self.rewards_history = []
    
    def log_episode(self, episode, steps, reward, epsilon, loss, duration):
        """Log episode statistics."""
        self.rewards_history.append(reward)
        
        # Calculate running average
        avg_reward = np.mean(self.rewards_history[-100:])
        
        with open(self.log_file, 'a') as f:
            f.write(f"{episode},{steps},{reward:.2f},{avg_reward:.2f},{epsilon:.4f},{loss:.6f},{duration:.2f}\n")
        
        return avg_reward


def train():
    """Main training loop."""
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
    logger = TrainingLogger()
    
    # Training parameters
    num_episodes = 500
    max_steps_per_episode = 3000
    save_interval = 50  # Save model every N episodes
    
    # Create save directory
    save_dir = "save/rl_models"
    os.makedirs(save_dir, exist_ok=True)
    
    best_reward = float('-inf')
    total_steps = 0
    start_episode = 1
    
    # Check for resume flag
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        checkpoint_path = os.path.join(save_dir, "checkpoint.json")
        model_path = os.path.join(save_dir, "latest_model.keras")
        
        if os.path.exists(model_path) and os.path.exists(checkpoint_path):
            print("\n🔄 Resuming from checkpoint...")
            agent.model = keras.models.load_model(model_path)
            agent.update_target_model()
            
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
            start_episode = checkpoint.get('episode', 1) + 1
            best_reward = checkpoint.get('best_reward', float('-inf'))
            total_steps = checkpoint.get('total_steps', 0)
            
            print(f"  Loaded model from: {model_path}")
            print(f"  Resuming at episode: {start_episode}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Best reward so far: {best_reward:.2f}")
        else:
            print("⚠️  No checkpoint found, starting fresh...")
    
    print(f"\nStarting training for {num_episodes} episodes (from episode {start_episode})...")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("-" * 60)
    
    # Live metrics tracking
    metrics = TrainingMetrics()
    log_interval = 100  # Print to terminal every N steps
    
    for episode in range(start_episode, num_episodes + 1):
        # Reset environment
        state_dict = game.reset_game()
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
            
            # Update live metrics
            metrics.step = step + 1
            metrics.total_steps = total_steps
            metrics.episode_reward = episode_reward
            metrics.epsilon = agent.epsilon
            metrics.loss = episode_loss / max(loss_count, 1)
            metrics.speed = next_state_dict['speed']
            metrics.action = int(action)
            metrics.memory_size = len(agent.memory)
            metrics.avg_reward_100 = np.mean(logger.rewards_history[-100:]) if logger.rewards_history else 0
            
            # Periodic terminal logging
            if total_steps % log_interval == 0:
                print(f"  [Step {total_steps:7d}] "
                      f"Ep {episode} | "
                      f"Step {step + 1:4d} | "
                      f"Reward: {episode_reward:7.1f} | "
                      f"Speed: {metrics.speed:5.0f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Mem: {len(agent.memory):5d}")
            
            if done:
                break
        
        # Episode complete
        duration = time.time() - start_time
        avg_loss = episode_loss / max(loss_count, 1)
        
        # Log episode
        avg_reward = logger.log_episode(
            episode, step + 1, episode_reward, 
            agent.epsilon, avg_loss, duration
        )
        
        # Print progress
        print(f"Episode {episode:4d} | "
              f"Steps: {step + 1:5d} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Avg(100): {avg_reward:8.2f} | "
              f"ε: {agent.epsilon:.4f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Time: {duration:.1f}s")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_full_model(os.path.join(save_dir, "best_model.keras"))
        
        # Periodic save
        if episode % save_interval == 0:
            agent.save_full_model(os.path.join(save_dir, f"model_ep{episode}.keras"))
            print(f"  → Checkpoint saved at episode {episode}")
        
        # Always save latest model and checkpoint for resume
        agent.model.save(os.path.join(save_dir, "latest_model.keras"), overwrite=True)
        checkpoint = {
            'episode': episode,
            'epsilon': agent.epsilon,
            'best_reward': best_reward,
            'total_steps': total_steps,
        }
        with open(os.path.join(save_dir, "checkpoint.json"), 'w') as f:
            json.dump(checkpoint, f)
    
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

    import keras
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else "save/rl_models/best_model.keras"
        num_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        evaluate(model_path, num_eps)
    elif len(sys.argv) > 1 and sys.argv[1] == "resume":
        # Resume training mode
        train()
    else:
        # Fresh training mode
        train()
