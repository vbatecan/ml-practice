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
CAR_WIDTH = 20
CAR_HEIGHT = 40
MAX_SPEED = 10
ACCELERATION = 0.2
FRICTION = 0.05
ROTATION_SPEED = 4
BRAKE_STRENGTH = 0.4

# Track settings
TRACK_WIDTH = 200
# Huge world settings
NUM_POINTS = 30
MIN_RADIUS = 800
MAX_RADIUS = 1600
SMOOTHING_ITERATIONS = 5

# World bounds (just for reference, track determines actual size)
WORLD_PADDING = 500
