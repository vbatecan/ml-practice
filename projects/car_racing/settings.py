# Screen settings
WIDTH = 1280
HEIGHT = 720
FPS = 60
TITLE = "Procedural Car Racing"

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (100, 100, 100)
DARK_GREY = (50, 50, 50)
GRASS_GREEN = (30, 100, 30)

# Car settings
CAR_WIDTH = 15
CAR_HEIGHT = 42

# Physics Constants (Pixels per Second / Seconds)
MAX_SPEED = 1000          # Pixels/sec
ACCELERATION = 500       # Pixels/sec^2
BRAKE_STRENGTH = 1200    # Pixels/sec^2
FRICTION = 200           # Natural deceleration (rolling resistance)
DRAG = 0.00100            # Air resistance factor (v^2)
TURN_SPEED = 180         # Degrees/sec
STEERING_SMOOTHING = 8.0  # Smoothness factor for steering (higher = more responsive/faster)
THROTTLE_RAMP = 3.0       # How fast throttle builds up (0 to 1)
BRAKE_RAMP = 5.0          # How fast brakes build up
HUD_SPEED_FACTOR = 0.25   # Pixels/s to km/h
HUD_ACCEL_FACTOR = 0.0694 # Pixels/s^2 to m/s^2 (Approx based on speed scaling)
DRIFT_FACTOR = 8       # Higher = less drift (grip), Lower = more drift
HANDBRAKE_FRICTION = 1000 # High friction when handbraking
HANDBRAKE_DRIFT_FACTOR = 1.0 # Very slippery, high drift

# Track settings
TRACK_WIDTH = 150

# Huge world settings
NUM_POINTS = 35
MIN_RADIUS = 2000
MAX_RADIUS = 3000 
SMOOTHING_ITERATIONS = 5

# World bounds (just for reference, track determines actual size)
WORLD_PADDING = 500

# Radar Settings
RADAR_ANGLES = [180, -135, -90, -45, -20, 0, 20, 45, 90, 135]
RADAR_COUNT = len(RADAR_ANGLES)         # Number of sensors
RADAR_MAX_DIST = 500    # Pixels
RADAR_STEP = 16          # Raycast precision step
RADAR_WARNING_DIST = 100
RADAR_DANGER_DIST = 50

# Visualization Settings
DRAW_RADARS = True
RADAR_VIEW_ONLY = False
FLASHLIGHT_MODE = False
FLASHLIGHT_RADIUS = 300
REALISTIC_VISION = True

# Model Settings
# MODEL_PLAYING = False
MODEL_PLAYING = True
# MODEL_PATH = "../car_racing_ml/model_2.h5"
MODEL_PATH = "../car_racing_ml/transferred.h5"
DATA_PATH = "../car_racing_ml/data.csv"
AI_FLOAT_CONTROL = False

# RL Environment Settings
RL_HEADLESS = False          # Disable rendering during training for speed
RL_MAX_STEPS = 2000          # Maximum steps per episode
RL_FIXED_DT = 1/60           # Fixed timestep for simulation stability

# RL Reward Tuning
RL_REWARD_COLLISION = -200   # Collision penalty
RL_REWARD_SPEED_FACTOR = 0.01     # Speed reward multiplier
RL_REWARD_CTE_FACTOR = 0.05       # Cross-track error penalty multiplier
RL_REWARD_HEADING_FACTOR = 0.1    # Heading error penalty multiplier
RL_REWARD_PROGRESS_FACTOR = 0.5   # Progress reward multiplier
RL_REWARD_LAP_BONUS = 100         # Bonus for completing a lap
