"""
Custom Gymnasium Environment for Car Racing RL.

This environment wraps the existing car racing game into a Gymnasium-compatible
format for training RL agents.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

from settings import *
from car import Car
from track import Track
from utils import get_closest_point_on_path, calculate_cross_track_error, calculate_heading_error


# Action mapping: 18 discrete actions
# Index -> (W, A, S, D, Handbrake)
ACTION_MAP = [
    (0, 0, 0, 0, 0),  # 0: No action
    (1, 0, 0, 0, 0),  # 1: Forward
    (0, 0, 1, 0, 0),  # 2: Brake
    (0, 1, 0, 0, 0),  # 3: Left only
    (0, 0, 0, 1, 0),  # 4: Right only
    (1, 1, 0, 0, 0),  # 5: Forward + Left
    (1, 0, 0, 1, 0),  # 6: Forward + Right
    (0, 1, 1, 0, 0),  # 7: Brake + Left
    (0, 0, 1, 1, 0),  # 8: Brake + Right
    # With Handbrake
    (0, 0, 0, 0, 1),  # 9: Handbrake only
    (1, 0, 0, 0, 1),  # 10: Forward + Handbrake
    (0, 0, 1, 0, 1),  # 11: Brake + Handbrake
    (0, 1, 0, 0, 1),  # 12: Left + Handbrake
    (0, 0, 0, 1, 1),  # 13: Right + Handbrake
    (1, 1, 0, 0, 1),  # 14: Forward + Left + Handbrake
    (1, 0, 0, 1, 1),  # 15: Forward + Right + Handbrake
    (0, 1, 1, 0, 1),  # 16: Brake + Left + Handbrake
    (0, 0, 1, 1, 1),  # 17: Brake + Right + Handbrake
]


class CarRacingEnv(gym.Env):
    """
    A Gymnasium environment for car racing.
    
    Observation Space (16 dimensions):
        - Speed: [0, MAX_SPEED]
        - SteeringAngle: [-360, 360]
        - CTE (Cross-Track Error): [-500, 500]
        - HeadingError: [-180, 180]
        - Radars R0-R9: [0, RADAR_MAX_DIST] x 10
    
    Action Space (Discrete, 18 actions):
        Combinations of WASD + Handbrake inputs.
    
    Reward:
        - Speed bonus
        - CTE penalty
        - Heading error penalty
        - Progress reward
        - Large collision penalty
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode=None, max_steps=2000, fixed_dt=1/60):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.fixed_dt = fixed_dt
        
        # Observation space: 16 continuous features
        # [Speed, SteeringAngle, Collision, Acceleration, CTE, HeadingError, R0, R1, ..., R9]
        obs_low = np.array([
            0,      # Speed min
            -360,   # SteeringAngle min
            0,      # Collision min
            -1000,  # Acceleration min
            -500,   # CTE min
            -180,   # HeadingError min
        ] + [0] * RADAR_COUNT, dtype=np.float32)
        
        obs_high = np.array([
            MAX_SPEED,         # Speed max
            360,               # SteeringAngle max
            1,                 # Collision max
            1000,              # Acceleration max
            500,               # CTE max
            180,               # HeadingError max
        ] + [RADAR_MAX_DIST] * RADAR_COUNT, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: 18 discrete actions
        self.action_space = spaces.Discrete(len(ACTION_MAP))
        
        # Initialize pygame only if rendering
        self._pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None
        
        # Game state
        self.track = None
        self.car = None
        self.current_step = 0
        self.last_closest_idx = 0
        self.total_progress = 0.0
        
        # Telemetry
        self.cte = 0.0
        self.heading_error = 0.0
        self.training_info = {} # Store info like episode, epsilon, etc.
        
    def _init_pygame(self):
        """Initialize pygame for rendering."""
        if not self._pygame_initialized:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Car Racing RL Environment")
            else:
                self.screen = pygame.Surface((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)
            self._pygame_initialized = True
    
    def _get_observation(self):
        """Extract observation from current game state."""
        speed = self.car.speed
        steering_angle = self.car.angle
        
        # Calculate navigation metrics
        closest_idx, _ = get_closest_point_on_path(self.car.pos, self.track.path)
        self.cte = calculate_cross_track_error(self.car.pos, self.track.path, closest_idx)
        self.heading_error = calculate_heading_error(
            self.car.angle, self.car.pos, self.track.path, closest_idx, lookahead=15
        )
        
        # Get radar distances
        radars = []
        if len(self.car.radars) == RADAR_COUNT:
            for r in self.car.radars:
                radars.append(r[2])  # Distance
        else:
            radars = [0] * RADAR_COUNT  # Fallback
        
        obs = np.array([
            speed,
            steering_angle,
            1.0 if self.car.collided else 0.0,
            self.car.instant_acceleration,
            self.cte,
            self.heading_error,
        ] + radars, dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self):
        """
        Calculate reward based on current state.
        
        Reward formula designed to encourage:
        - High speed (exploration beyond human data)
        - Staying near track center (low CTE)
        - Facing the right direction (low heading error)
        - Forward progress on track
        - Avoiding collisions
        """
        if self.car.collided:
            return -200.0  # Harsh collision penalty
        
        # Speed reward: Encourage going fast (0 to ~10 points)
        speed_reward = self.car.speed * 0.01
        
        # CTE penalty: Stay on track center (0 to ~-25 at max deviation)
        cte_penalty = -abs(self.cte) * 0.05
        
        # Heading penalty: Face the right direction (0 to ~-18 at 180° error)
        heading_penalty = -abs(self.heading_error) * 0.1
        
        # Progress reward: Track forward movement on the racing line
        closest_idx, _ = get_closest_point_on_path(self.car.pos, self.track.path)
        path_len = len(self.track.path)
        
        # Calculate progress (handling wraparound)
        progress_delta = closest_idx - self.last_closest_idx
        if progress_delta < -path_len // 2:  # Wrapped around
            progress_delta += path_len
        elif progress_delta > path_len // 2:  # Went backwards
            progress_delta -= path_len
        
        progress_reward = max(0, progress_delta) * 0.5  # Only reward forward progress
        
        self.last_closest_idx = closest_idx
        self.total_progress += max(0, progress_delta)
        
        total_reward = speed_reward + cte_penalty + heading_penalty + progress_reward
        
        return total_reward
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize pygame if needed
        if self.render_mode is not None:
            self._init_pygame()
        
        # Create new track and car
        self.track = Track()
        self.car = Car(self.track.start_position)
        
        # Reset state
        self.current_step = 0
        self.last_closest_idx = 0
        self.total_progress = 0.0
        self.cte = 0.0
        self.heading_error = 0.0
        
        # Initial sensor update
        self.car.update(self.fixed_dt, self.track.mask, (0, 0, 0, 0, 0))
        
        obs = self._get_observation()
        info = {"total_progress": 0.0}
        
        return obs, info
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Discrete action index (0-17)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action to WASD + Handbrake
        controls = ACTION_MAP[action]
        
        # Update game state
        self.car.update(self.fixed_dt, self.track.mask, controls)
        self.current_step += 1
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self.car.collided  # Episode ends on collision
        truncated = self.current_step >= self.max_steps  # Max steps reached
        
        # Lap completion check (optional bonus)
        if self.total_progress >= len(self.track.path):
            reward += 100.0  # Lap completion bonus
            terminated = True
        
        info = {
            "total_progress": self.total_progress,
            "speed": self.car.speed,
            "cte": self.cte,
            "heading_error": self.heading_error,
            "step": self.current_step,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state."""
        if self.render_mode is None:
            return None
        
        self._init_pygame()
        
        # Camera offset (center on car)
        camera_x = self.car.rect.centerx - WIDTH // 2
        camera_y = self.car.rect.centery - HEIGHT // 2
        camera_offset = pygame.math.Vector2(camera_x, camera_y)
        
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw track
        track_pos = self.track.rect.topleft - camera_offset
        self.screen.blit(self.track.image, track_pos)
        
        # Draw car
        car_pos = self.car.rect.topleft - camera_offset
        self.screen.blit(self.car.image, car_pos)
        
        # Draw radars
        for start, end, dist in self.car.radars:
            start_off = start - camera_offset
            end_off = end - camera_offset
            
            line_color = GREEN
            if dist < RADAR_DANGER_DIST:
                line_color = RED
            elif dist < RADAR_WARNING_DIST:
                line_color = (255, 255, 0)  # Yellow
            
            pygame.draw.line(self.screen, line_color, start_off, end_off, 1)
            pygame.draw.circle(self.screen, line_color, (int(end_off.x), int(end_off.y)), 3)
        
        # Draw HUD
        self._draw_hud()
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
        return None
    
    def _draw_hud(self):
        """Draw heads-up display."""
        # Semi-transparent panel
        hud_surf = pygame.Surface((250, 150), pygame.SRCALPHA)
        pygame.draw.rect(hud_surf, (0, 0, 0, 128), hud_surf.get_rect(), border_radius=10)
        self.screen.blit(hud_surf, (10, 10))
        
        # Speed
        display_speed = int(self.car.speed * 0.25)
        speed_color = GREEN if display_speed < 100 else (RED if display_speed > 160 else WHITE)
        speed_surf = self.font.render(f"Speed: {display_speed} km/h", True, speed_color)
        self.screen.blit(speed_surf, (20, 20))
        
        # CTE
        cte_surf = self.font.render(f"CTE: {int(self.cte)}", True, WHITE)
        self.screen.blit(cte_surf, (20, 50))
        
        # Heading
        head_surf = self.font.render(f"Heading: {int(self.heading_error)}°", True, WHITE)
        self.screen.blit(head_surf, (20, 80))
        
        # Step / Progress
        step_surf = self.font.render(f"Step: {self.current_step}/{self.max_steps}", True, GREY)
        self.screen.blit(step_surf, (20, 110))

        # Training Info (if provided)
        if self.training_info:
            start_y = 10
            for label, value in self.training_info.items():
                if isinstance(value, float):
                    text = f"{label}: {value:.4f}"
                else:
                    text = f"{label}: {value}"
                
                info_surf = self.font.render(text, True, (200, 200, 255))
                # Draw on the top right
                self.screen.blit(info_surf, (WIDTH - 200, start_y))
                start_y += 30
    
    def close(self):
        """Clean up resources."""
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
            self.screen = None


# Register the environment with Gymnasium
def register_env():
    """Register CarRacingEnv with Gymnasium."""
    gym.register(
        id="CarRacing-v0",
        entry_point="env:CarRacingEnv",
        max_episode_steps=2000,
    )


if __name__ == "__main__":
    # Quick test
    print("Testing CarRacingEnv...")
    
    env = CarRacingEnv(render_mode="human")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    for i in range(500):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {i+1}, reward: {reward}, info: {info}")
            obs, info = env.reset()
    
    env.close()
    print("Test completed!")
