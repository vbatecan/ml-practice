import pygame

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
MAX_SPEED = 800          # Pixels/sec
ACCELERATION = 600       # Pixels/sec^2
BRAKE_STRENGTH = 1200    # Pixels/sec^2
FRICTION = 200           # Natural deceleration (rolling resistance)
DRAG = 0.00015            # Air resistance factor (v^2)
TURN_SPEED = 180         # Degrees/sec
DRIFT_FACTOR = 5       # Higher = less drift (grip), Lower = more drift

# Track settings
TRACK_WIDTH = 200

# Huge world settings
NUM_POINTS = 30
MIN_RADIUS = 1000
MAX_RADIUS = 2500
SMOOTHING_ITERATIONS = 5

# World bounds (just for reference, track determines actual size)
WORLD_PADDING = 500

# Radar Settings
RADAR_COUNT = 5          # Number of sensors
RADAR_MAX_DIST = 2000    # Pixels
RADAR_STEP = 10          # Raycast precision step
