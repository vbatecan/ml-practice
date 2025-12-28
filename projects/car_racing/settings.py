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
MAX_SPEED = 1000          # Pixels/sec
ACCELERATION = 500       # Pixels/sec^2
BRAKE_STRENGTH = 1200    # Pixels/sec^2
FRICTION = 200           # Natural deceleration (rolling resistance)
DRAG = 0.00100            # Air resistance factor (v^2)
TURN_SPEED = 180         # Degrees/sec
DRIFT_FACTOR = 32       # Higher = less drift (grip), Lower = more drift

# Track settings
TRACK_WIDTH = 230

# Huge world settings
NUM_POINTS = 40
MIN_RADIUS = 1000
MAX_RADIUS = 4000
SMOOTHING_ITERATIONS = 7

# World bounds (just for reference, track determines actual size)
WORLD_PADDING = 500

# Radar Settings
RADAR_COUNT = 8         # Number of sensors
RADAR_MAX_DIST = 500    # Pixels
RADAR_STEP = 16          # Raycast precision step
RADAR_WARNING_DIST = 100
RADAR_DANGER_DIST = 50

# Visualization Settings
DRAW_RADARS = True
RADAR_VIEW_ONLY = False
FLASHLIGHT_MODE = False
FLASHLIGHT_RADIUS = 300

# Model Settings
MODEL_PLAYING = False
# MODEL_PLAYING = True
# MODEL_PATH = "../car_racing_ml/model.h5"
MODEL_PATH = "../car_racing_ml/transferred.h5"
DATA_PATH = "../car_racing_ml/data.csv"
