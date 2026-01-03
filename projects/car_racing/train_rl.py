"""
Reinforcement Learning Training Script for Car Racing (SAC)
Implements SAC training loop with:
- Continuous action space
- Reward shaping
- Model checkpointing (Resume support)
- Graceful shutdown
"""

import json
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from agent import SACAgent
from main import Game
from settings import FPS, RADAR_COUNT, RL_HEADLESS

# Checkpoint paths
CHECKPOINT_DIR = "save/rl_models"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.json")
# We use a prefix for SAC weights
SAC_WEIGHTS_PREFIX = os.path.join(CHECKPOINT_DIR, "sac_checkpoint")
LATEST_MODEL_FILE = os.path.join(CHECKPOINT_DIR, "latest_model.keras") # Inference model
MEMORY_FILE = os.path.join(CHECKPOINT_DIR, "replay_memory.pkl")

# Global reference for signal handler
_current_agent = None
_current_episode = 0
_best_reward = float('-inf')
_total_steps = 0


def graceful_shutdown(signum, frame):
    """Handle Ctrl+C gracefully by saving state before exit."""
    global _current_agent, _current_episode, _best_reward, _total_steps
    print("\n\n⚠️  Interrupt received! Saving state...")
    
    if _current_agent is not None:
        # Save inference model
        _current_agent.save_full_model(LATEST_MODEL_FILE)
        # Save full training state
        _current_agent.save_models(SAC_WEIGHTS_PREFIX)
        # Save memory
        _current_agent.save_memory(MEMORY_FILE)
        # Save checkpoint
        save_checkpoint(_current_episode, _best_reward, _total_steps)
        print("✓ State saved successfully. Safe to exit.")
    
    sys.exit(0)


def save_checkpoint(episode, best_reward, total_steps):
    """Save training checkpoint metadata."""
    checkpoint = {
        "episode": episode,
        "best_reward": best_reward,
        "total_steps": total_steps,
    }
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def load_checkpoint():
    """Load training checkpoint metadata."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return None


def preprocess_state(state_dict):
    """Convert game state dictionary to normalized feature vector."""
    # 1. Log-scale and normalize radars
    radars = np.array(state_dict["radars"], dtype=np.float32)
    radars_log = np.log1p(radars) / 7.0  # log(1000) ≈ 7

    # 2. Normalize continuous features
    speed = state_dict["speed"] / 2000.0  # MAX_SPEED (approx)
    angle = state_dict["angle"] / 360.0  # Normalize to typically [-1, 1] range
    acceleration = np.clip(state_dict["acceleration"] / 500.0, -1, 1)
    cte = np.clip(state_dict["cte"] / 100.0, -1, 1)

    # 3. Combine into feature vector
    # Order: [speed, angle, acceleration, cte, R0...Rn]
    base_features = np.array(
        [speed, angle, acceleration, cte], dtype=np.float32
    )
    return np.concatenate([base_features, radars_log])


# Track slow time for progressive penalty
_slow_steps_counter = 0
_SLOW_THRESHOLD = 30
_GRACE_PERIOD_STEPS = 120


def reset_slow_counter():
    global _slow_steps_counter
    _slow_steps_counter = 0


def calculate_reward(state_dict, prev_state_dict, action_vector):
    """
    Calculate shaped reward based on driving behavior.
    """
    global _slow_steps_counter
    reward = 0.0

    # === TERMINAL CONDITION: COLLISION ===
    if state_dict["collided"]:
        return -500.0

    # === PROGRESS REWARD ===
    speed = state_dict["speed"]
    
    if speed < _SLOW_THRESHOLD:
        _slow_steps_counter += 1
    else:
        _slow_steps_counter = 0
    
    # Reward for speed
    if speed > _SLOW_THRESHOLD:
        reward += speed * 0.003
    
    # === TIME-BASED SLOW PENALTY ===
    if speed < _SLOW_THRESHOLD:
        if _slow_steps_counter > _GRACE_PERIOD_STEPS:
            overtime_seconds = (_slow_steps_counter - _GRACE_PERIOD_STEPS) / FPS
            slow_penalty = min(0.3 + overtime_seconds * 0.1, 4.0)
            reward -= slow_penalty
        else:
            reward -= 0.3

    # === TRACK ALIGNMENT ===
    cte = abs(state_dict["cte"])
    if cte < 35:
        reward += 0.05
    elif cte > 35:
        reward -= cte * 0.01

    # === WALL PROXIMITY PENALTY ===
    radars = state_dict["radars"]
    danger_threshold = 50 
    warning_threshold = 100
    
    for radar_dist in radars:
        if radar_dist < danger_threshold:
            reward -= 0.3
        elif radar_dist < warning_threshold:
            proximity_penalty = (warning_threshold - radar_dist) / warning_threshold
            reward -= proximity_penalty * 0.2

    # === CHECKPOINT PROGRESS BONUS ===
    # Small bonus when checkpoint is crossed (tracked externally)
    if state_dict.get("crossed_checkpoint", False):
        reward += 0.02  # Small continuous bonus for being past checkpoint

    return reward


def map_continuous_to_controls(action_vector):
    """
    Map SAC continuous action vector to game controls.
    Vector: [steering (-1 to 1), throttle_brake (-1 to 1)]
    """
    steering = float(action_vector[0])
    accel = float(action_vector[1])

    # Steering (A/D)
    # A is Left (positive target turning in car.py uses A?)
    # car.py: if keys[pygame.K_a]: target_turning = 1 (Left)
    # So we map positive steering to A.
    if steering > 0:
        a = steering
        d = 0.0
    else:
        a = 0.0
        d = -steering

    # Throttle/Brake (W/S)
    if accel > 0:
        w = accel
        s = 0.0
    else:
        w = 0.0
        s = -accel

    return (w, a, s, d, 0.0) # hb = 0


class TrainingMetrics:
    def __init__(self):
        self.episode = 0
        self.step = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.alpha_loss = 0.0
        self.critic_loss = 0.0
        self.actor_loss = 0.0
        self.speed = 0.0
        self.action = [0.0, 0.0]
        self.memory_size = 0
        self.avg_reward_100 = 0.0
        self.alpha = 0.0
        # Lap tracking
        self.current_lap = 0
        self.total_laps = 0
        self.lap_reward = 0.0

    def to_dict(self):
        return {
            "episode": self.episode,
            "step": self.step,
            "total_steps": self.total_steps,
            "episode_reward": self.episode_reward,
            "loss": self.critic_loss, # General loss for simple HUD
            "speed": self.speed,
            "action": 0, # Placeholder for HUD index
            "memory_size": self.memory_size,
            "avg_reward_100": self.avg_reward_100,
            "epsilon": self.alpha, # Reusing epsilon field for Alpha in HUD
            "current_lap": self.current_lap,
            "total_laps": self.total_laps,
        }


class TrainingLogger:
    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_sac_{timestamp}.csv")
        with open(self.log_file, "w") as f:
            f.write("episode,steps,reward,avg_reward_100,alpha,critic_loss,actor_loss,duration\n")
        self.rewards_history = []

    def log_episode(self, episode, steps, reward, alpha, critic_loss, actor_loss, duration):
        self.rewards_history.append(reward)
        avg_reward = np.mean(self.rewards_history[-100:])
        with open(self.log_file, "a") as f:
            f.write(
                f"{episode},{steps},{reward:.2f},{avg_reward:.2f},{alpha:.4f},{critic_loss:.6f},{actor_loss:.6f},{duration:.2f}\n"
            )
        return avg_reward


def train(resume=False):
    global _current_agent, _current_episode, _best_reward, _total_steps
    
    signal.signal(signal.SIGINT, graceful_shutdown)
    
    print("=" * 60)
    print("SAC Training for Car Racing")
    print("=" * 60)

    game = Game()
    initial_state = game.reset_game()
    state_size = 4 + RADAR_COUNT
    action_size = 2 # [Steering, Throttle/Brake]

    print(f"State size: {state_size}")
    print(f"Action size: {action_size} (Continuous)")
    
    agent = SACAgent(state_size, action_size)
    _current_agent = agent
    
    logger = TrainingLogger()
    metrics = TrainingMetrics()

    num_episodes = 2000
    max_steps_per_episode = 2000
    save_interval = 50
    
    # Action repeat helps with continuous control too, but SAC controls are smooth.
    # We can use a small repeat or 1.
    ACTION_REPEAT = 2 

    save_dir = CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)

    best_reward = float("-inf")
    total_steps = 0
    start_episode = 1

    if resume:
        checkpoint = load_checkpoint()
        # Check if weights exist
        if checkpoint and os.path.exists(f"{SAC_WEIGHTS_PREFIX}_actor.weights.h5"):
            start_episode = checkpoint["episode"] + 1
            best_reward = checkpoint["best_reward"]
            total_steps = checkpoint["total_steps"]
            
            print("Loading weights...")
            if agent.load_models(SAC_WEIGHTS_PREFIX):
                print("✓ Weights loaded.")
                # Try memory
                if agent.load_memory(MEMORY_FILE):
                    print(f"✓ Memory loaded ({len(agent.memory_buffer)} items).")
                else:
                    print("  Memory not found or failed.")
            else:
                print("  Failed to load weights. Starting fresh.")
        else:
            print("No complete checkpoint found. Starting fresh.")

    _best_reward = best_reward
    _total_steps = total_steps

    print(f"\nStarting training from episode {start_episode}...")
    print("-" * 60)

    for episode in range(start_episode, num_episodes + 1):
        state_dict = game.reset_game()
        reset_slow_counter()
        state = preprocess_state(state_dict)

        episode_reward = 0.0
        episode_critic_loss = 0.0
        episode_actor_loss = 0.0
        train_steps = 0
        
        start_time = time.time()
        metrics.episode = episode

        for step in range(max_steps_per_episode):
            # Sample action
            action = agent.sample_action(state)
            
            controls = map_continuous_to_controls(action)
            
            accumulated_reward = 0.0
            next_state_dict = state_dict
            
            for _ in range(ACTION_REPEAT):
                next_state_dict, _, done, info = game.step(controls, metrics.to_dict())
                step_reward = calculate_reward(next_state_dict, state_dict, action)
                # Add lap rewards from the game
                lap_reward = info.get("lap_reward", 0.0)
                step_reward += lap_reward
                accumulated_reward += step_reward
                state_dict = next_state_dict # Update for prev_state calculations in next repeat (approx)
                if done: break
            
            next_state = preprocess_state(next_state_dict)
            
            agent.remember(state, action, accumulated_reward, next_state, done)

            # Train every step (or every N steps)
            loss_info = agent.train()
            if loss_info != 0.0:
                # Expect loss_info to be tensor or float
                episode_critic_loss += float(loss_info) # Simplified
                train_steps += 1

            state = next_state
            episode_reward += accumulated_reward
            total_steps += 1
            _total_steps = total_steps

            # Update metrics
            metrics.step = step
            metrics.total_steps = total_steps
            metrics.episode_reward = episode_reward
            metrics.speed = next_state_dict["speed"]
            metrics.alpha = float(agent.alpha)
            metrics.memory_size = len(agent.memory_buffer)
            metrics.avg_reward_100 = (np.mean(logger.rewards_history[-100:]) if logger.rewards_history else 0)
            metrics.current_lap = next_state_dict.get("current_lap", 0)
            metrics.total_laps = next_state_dict.get("total_laps", 0)

            if total_steps % 100 == 0:
                print(f"  Ep {episode} | Step {step} | Rw {episode_reward:.1f} | α {metrics.alpha:.3f}")

            if done:
                break

        duration = time.time() - start_time
        avg_critic_loss = episode_critic_loss / max(train_steps, 1)
        
        avg_reward = logger.log_episode(
            episode, step, episode_reward, float(agent.alpha), avg_critic_loss, 0, duration
        )

        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | α: {float(agent.alpha):.3f} | Time: {duration:.1f}s")

        # Save Best
        if episode_reward > best_reward:
            best_reward = episode_reward
            _best_reward = best_reward
            agent.save_full_model(os.path.join(save_dir, "best_model.keras"))
            print("  ★ New Best Reward!")

        _current_episode = episode
        
        # Save Regular checkpoints
        agent.save_full_model(LATEST_MODEL_FILE) # For Game
        agent.save_models(SAC_WEIGHTS_PREFIX) # For Resume
        save_checkpoint(episode, best_reward, total_steps)
        
        if episode % 10 == 0:
            agent.save_memory(MEMORY_FILE)
            
        if episode % save_interval == 0:
            agent.save_full_model(os.path.join(save_dir, f"model_ep{episode}.keras"))

    print("Training Complete.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "resume":
        train(resume=True)
    else:
        train(resume=False)
