import numpy as np

from agent import DQNAgent
from main import Game
from settings import *


def preprocess_state(state_dict):
    # 1. Handle Radars (Log scale + Normalization)
    # Most radars in simulators max out at a certain distance (e.g., 1000)
    radars = np.array(state_dict["radars"], dtype=np.float32)
    radars_log = np.log1p(radars) / 7.0  # Dividing by ~7 flattens log(1000) to ~1.0

    # 2. Handle Heading Error and CTE
    # These can be positive or negative; divide by a reasonable max to keep them near [-1, 1]
    heading = state_dict["heading_error"] / 180.0
    cte = state_dict["cte"] / 5.0  # Assuming 5.0 is the track width limit

    # 3. Handle Speed and Acceleration
    speed = state_dict["speed"] / 100.0  # Normalize by max speed of your game

    # 4. Combine into a single flat array
    # Vector: [speed, angle, acceleration, cte, heading, R0...R9]
    state_vector = np.array(
        [
            speed,
            state_dict["angle"] / 360.0,
            state_dict["acceleration"] / 10.0,
            cte,
            heading,
        ]
    )

    # Concatenate the radars onto the end
    return np.concatenate([state_vector, radars_log])


def calculate_reward(state_dict, prev_distance):
    # 1. Reward for progress
    reward = state_dict["total_distance"] - prev_distance

    # 2. Penalty for being far from center (CTE) or facing wrong way
    reward -= abs(state_dict["cte"]) * 0.1
    reward -= abs(state_dict["heading_error"]) * 0.01

    # 3. Massive penalty for collision
    if state_dict["collided"]:
        reward = -100

    return reward


def map_action_to_controls(action_idx):
    """
    Maps a single integer from the DQN (0-4) to 
    the (w, a, s, d) format your input function expects.
    """
    # Format: (w, a, s, d)
    mapping = {
        0: [1, 0, 0, 0], # Hard Forward (Full Throttle)
        1: [1, 1, 0, 0], # Forward + Left
        2: [1, 0, 0, 1], # Forward + Right
        3: [0, 0, 1, 0], # Brake/Reverse
        4: [0, 0, 0, 0]  # Coast
    }
    return mapping.get(action_idx, [0, 0, 0, 0])


def main():
    game = Game()
    initial_state = game.get_state()
    state_size = len(initial_state["radars"]) + 5
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    episodes = 50
    max_steps = 5000

    for e in range(episodes):
        state = game.reset_game()

        for steps in range(max_steps):
            action = agent.act(state)
            controls = map_action_to_controls(action)
            next_state, reward, done, info = game.step(controls)

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                break


if __name__ == "__main__":
    main()
