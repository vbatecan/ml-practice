# Car Racing Game API Documentation

This document describes the programmatic API for interacting with the car racing game, designed for reinforcement learning and automated control systems.

## Quick Start

```python
from main import Game

# Initialize the game
game = Game()

# Reset to get initial state
state = game.reset_game()

# Game loop
done = False
while not done:
    action = (1, 0, 0, 0, 0)  # Example: accelerate forward
    state, done, info = game.step(action)
    
    # Calculate your own reward based on state
    reward = your_reward_function(state)
```

---

## Methods

### `Game.__init__()`

Initializes the game environment.

**Behavior:**
- If `RL_HEADLESS` is `True` in settings, runs without display (faster for training)
- Loads the track and initializes the car at the start position
- If `MODEL_PLAYING` is enabled, loads a pre-trained model for inference

---

### `reset_game() -> dict`

Resets the game to its initial state.

**Returns:** `state` dictionary (see [State Dictionary](#state-dictionary))

**Example:**
```python
state = game.reset_game()
```

**Notes:**
- Generates a new random track
- Resets car position, velocity, and angle
- Clears collision state and statistics
- Updates sensors immediately so the returned state is valid

---

### `step(action) -> tuple[dict, bool, dict]`

Advances the game by one frame with the given action.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `action` | `tuple` | 5-element tuple: `(w, a, s, d, handbrake)` |

**Action Format:**
| Index | Key | Description | Values |
|-------|-----|-------------|--------|
| 0 | `w` | Accelerate | `0` or `1` (or `0.0`-`1.0` for analog) |
| 1 | `a` | Steer Left | `0` or `1` (or `0.0`-`1.0` for analog) |
| 2 | `s` | Brake/Reverse | `0` or `1` (or `0.0`-`1.0` for analog) |
| 3 | `d` | Steer Right | `0` or `1` (or `0.0`-`1.0` for analog) |
| 4 | `hb` | Handbrake | `0` or `1` |

**Returns:** `(state, done, info)`
| Return | Type | Description |
|--------|------|-------------|
| `state` | `dict` | Current game state (see [State Dictionary](#state-dictionary)) |
| `done` | `bool` | `True` if episode terminated (collision) |
| `info` | `dict` | Additional info (currently empty, reserved for future use) |

**Example:**
```python
# Accelerate and steer right
action = (1, 0, 0, 1, 0)
state, done, info = game.step(action)
```

---

### `get_state() -> dict`

Returns the current game state without advancing the simulation.

**Returns:** `state` dictionary (see [State Dictionary](#state-dictionary))

**Example:**
```python
state = game.get_state()
```

---

## State Dictionary

The state dictionary contains all observables for the agent:

| Key | Type | Description |
|-----|------|-------------|
| `speed` | `float` | Current speed (internal units, ~0-1000) |
| `angle` | `float` | Car heading angle in degrees |
| `acceleration` | `float` | Instantaneous acceleration |
| `cte` | `float` | Cross-Track Error (distance from center line, negative=left, positive=right) |
| `heading_error` | `float` | Angle difference from ideal heading (degrees) |
| `radars` | `list[float]` | List of radar distances (length = `RADAR_COUNT` from settings) |
| `collided` | `bool` | `True` if car has collided with track boundary |
| `total_distance` | `float` | Total distance traveled in episode |

**Example State:**
```python
{
    'speed': 245.3,
    'angle': 45.2,
    'acceleration': 12.5,
    'cte': -23.4,
    'heading_error': 5.7,
    'radars': [150, 200, 300, 400, 350, 280, 180, 120, 90, 85],
    'collided': False,
    'total_distance': 15420.5
}
```

---

## Configuration (settings.py)

Key settings that affect the API:

| Setting | Default | Description |
|---------|---------|-------------|
| `RL_HEADLESS` | `False` | Run without display (set `True` for training) |
| `RL_FIXED_DT` | `0.016` | Fixed timestep for `step()` (~60 FPS equivalent) |
| `RADAR_COUNT` | `10` | Number of radar sensors |
| `RADAR_LENGTH` | `500` | Maximum radar detection distance |

---

## Example: Custom Reward Function

```python
def calculate_reward(state, done):
    if done:
        return -100  # Collision penalty
    
    reward = 0.0
    
    # Reward for speed
    reward += state['speed'] * 0.01
    
    # Penalty for being off-center
    reward -= abs(state['cte']) * 0.001
    
    # Penalty for wrong heading
    reward -= abs(state['heading_error']) * 0.01
    
    # Survival bonus
    reward += 1.0
    
    return reward
```

---

## Example: Full Training Loop

```python
from main import Game

game = Game()

for episode in range(1000):
    state = game.reset_game()
    total_reward = 0
    
    while True:
        # Your agent selects an action
        action = agent.select_action(state)
        
        # Step the environment
        next_state, done, info = game.step(action)
        
        # Calculate reward
        reward = calculate_reward(next_state, done)
        total_reward += reward
        
        # Store transition for learning
        agent.store(state, action, reward, next_state, done)
        
        state = next_state
        
        if done:
            break
    
    print(f"Episode {episode}: Total Reward = {total_reward}")
```
