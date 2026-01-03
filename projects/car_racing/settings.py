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
CAR_WIDTH = 22
CAR_HEIGHT = 42

# Physics Constants (Pixels per Second / Seconds)
MAX_SPEED = 1000  # Pixels/sec
ACCELERATION = 1000  # Pixels/sec^2
BRAKE_STRENGTH = 1200  # Pixels/sec^2
FRICTION = 200  # Natural deceleration (rolling resistance)
DRAG = 0.00100  # Air resistance factor (v^2)
TURN_SPEED = 180  # Degrees/sec
STEERING_SMOOTHING = (
    6.0  # Smoothness factor for steering (higher = more responsive/faster)
)
THROTTLE_RAMP = 5.0  # How fast throttle builds up (0 to 1)
BRAKE_RAMP = 5.0  # How fast brakes build up
HUD_SPEED_FACTOR = 0.25  # Pixels/s to km/h
HUD_ACCEL_FACTOR = 0.0694  # Pixels/s^2 to m/s^2 (Approx based on speed scaling)
DRIFT_FACTOR = 12  # Higher = less drift (grip), Lower = more drift
HANDBRAKE_FRICTION = 1000  # High friction when handbraking
HANDBRAKE_DRIFT_FACTOR = 1.0  # Very slippery, high drift

# Track settings
TRACK_WIDTH = 225

# Huge world settings
NUM_POINTS = 30
MIN_RADIUS = 1000
MAX_RADIUS = 2000
SMOOTHING_ITERATIONS = 8

# World bounds (just for reference, track determines actual size)
WORLD_PADDING = 500

# Radar Settings
RADAR_ANGLES = list(range(-130, 131, 30)) + [180]  # 28 sensors: -135, -125, ..., 135
RADAR_COUNT = len(RADAR_ANGLES)  # Number of sensors
RADAR_MAX_DIST = 1500  # Pixels
RADAR_STEP = 16  # Raycast precision step
RADAR_WARNING_DIST = 100
RADAR_DANGER_DIST = 50

# Visualization Settings
DRAW_RADARS = True
RADAR_VIEW_ONLY = False
FLASHLIGHT_MODE = False
FLASHLIGHT_RADIUS = 300
REALISTIC_VISION = True

# Model Settings
MODEL_PLAYING = False
# MODEL_PLAYING = True
MODEL_PATH = "../car_racing_ml/model_2.h5"
# MODEL_PATH = "../car_racing_ml/transferred.h5"
DATA_PATH = "../car_racing_ml/combined_data.csv"
AI_FLOAT_CONTROL = False

# RL Environment Settings
RL_HEADLESS = False  # Set to True for faster training, False to visualize
RL_FIXED_DT = 1 / FPS
REPLAY_MEMORY = 1000000  # Large replay buffer for experience diversity
BATCH_SIZE = 256  # Larger batch for stable gradients

# Telemetry Settings
TELEMETRY_CLEANUP_ENABLED = False
TELEMETRY_LOOKBACK_SECONDS = 2.0

# Lap System Settings
TOTAL_LAPS = 3  # Number of laps to complete a race
LAP_CHECKPOINT_INDEX = 0.5  # Position on track (0-1) for lap line (0.5 = opposite side)
LAP_BASE_REWARD = 100.0  # Base reward for completing a lap
LAP_FINAL_MULTIPLIER = 3.0  # Multiplier for finishing final lap
EXPECTED_LAP_TIME = 60.0  # Expected time in seconds for a decent lap
TIME_BONUS_MAX = 200.0  # Maximum time bonus for a fast lap
TIME_PENALTY_PER_SECOND = 2.0  # Penalty per second over expected time (capped)
