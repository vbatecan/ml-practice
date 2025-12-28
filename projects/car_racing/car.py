import pygame
import math
from settings import *

class Car(pygame.sprite.Sprite):
    def __init__(self, start_pos):
        super().__init__()
        # Create a simple car image
        self.original_image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        self.original_image.fill(BLUE) # Car color
        
        # Add visual details
        pygame.draw.rect(self.original_image, BLACK, (0, 0, CAR_WIDTH, 4)) # Front bumper
        pygame.draw.rect(self.original_image, WHITE, (2, 8, CAR_WIDTH-4, 8)) # Windshield
        
        self.image = self.original_image
        self.rect = self.image.get_rect(center=start_pos)
        
        self.pos = pygame.math.Vector2(start_pos)
        self.velocity = pygame.math.Vector2(0, 0)
        self.angle = 0
        self.speed = 0
        
        self.mask = pygame.mask.from_surface(self.image)
        
        # Input state
        self.turning = 0.0 # Actual smoothed turning value
        self.target_turning = 0.0 # Input target
        self.accelerating = 0 
        
        # Sensor/Radar
        self.radars = []
        self.last_speed = 0
        self.instant_acceleration = 0
        
        self.collided = False


    def update(self, dt, track_mask=None, controls=None):
        self.input(controls)
        self.move(dt)
        self.update_sensors(track_mask)
        self.check_collision(track_mask)

    def update_sensors(self, track_mask):
        self.radars.clear()
        if not track_mask: return
            
        radar_angles = []
        if RADAR_COUNT > 0:
            step_angle = 360 / RADAR_COUNT
            for i in range(RADAR_COUNT):
                radar_angles.append(i * step_angle)
        
        for angle_offset in radar_angles:
            check_angle = self.angle + angle_offset
            rad = math.radians(check_angle)
            
            dir_vec = pygame.math.Vector2(-math.sin(rad), -math.cos(rad))
            start_pos = pygame.math.Vector2(self.rect.center)
            current_pos = start_pos.copy()
            
            dist = 0
            step = RADAR_STEP
            
            while dist < RADAR_MAX_DIST:
                current_pos += dir_vec * step
                dist += step
                
                int_x, int_y = int(current_pos.x), int(current_pos.y)
                
                if 0 <= int_x < track_mask.get_size()[0] and 0 <= int_y < track_mask.get_size()[1]:
                    if track_mask.get_at((int_x, int_y)):
                        current_pos -= dir_vec * step
                        dist -= step
                        step = 1 # Fine step
                        while dist < RADAR_MAX_DIST:
                            current_pos += dir_vec * step
                            dist += step
                            int_x, int_y = int(current_pos.x), int(current_pos.y)
                            if track_mask.get_at((int_x, int_y)):
                                break
                            if step == 1 and dist > RADAR_MAX_DIST: break # safety
                        break
                else:
                    break
            
            self.radars.append((start_pos, current_pos, dist))

    def input(self, controls=None):
        self.target_turning = 0
        self.accelerating = 0
        
        if controls:
            w, a, s, d = controls
            # Analog steering
            if a or d:
                self.target_turning = float(a) - float(d)
                
            if w: self.accelerating = 1
            if s: self.accelerating = -1
            return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.target_turning = 1
        if keys[pygame.K_d]:
            self.target_turning = -1
            
        if keys[pygame.K_w]:
            self.accelerating = 1
        if keys[pygame.K_s]:
            self.accelerating = -1

    def move(self, dt):
        self.prev_pos = self.pos.copy()
        
        # Smooth steering
        # Lerp factor: Adjust for responsiveness. 
        # 5.0 * dt means it takes ~0.2s to reach target.
        smoothing_speed = 5.0 
        diff = self.target_turning - self.turning
        self.turning += diff * min(smoothing_speed * dt, 1.0)
        
        if abs(self.speed) > 0:
            # Scale turning by speed to avoid spinning in place
            # At 200 px/s ~ 1.0 factor
            steering_factor = abs(self.speed) / 250.0
            steering_factor = min(1.2, max(0.1, steering_factor)) # Clamp
            
            if abs(self.speed) < 10: steering_factor = 0 # Deadzone
            
            self.angle += self.turning * TURN_SPEED * steering_factor * dt

        rad = math.radians(self.angle)
        heading = pygame.math.Vector2(-math.sin(rad), -math.cos(rad))
        
        # 3. Acceleration
        if self.accelerating == 1:
            # Acceleration dampening ONLY when close to MAX_SPEED
            # This allows full acceleration for most of the range (0-70%)
            accel_factor = 1.0
            if self.speed > MAX_SPEED * 0.7:
                percent_over = (self.speed - (MAX_SPEED * 0.7)) / (MAX_SPEED * 0.3)
                accel_factor = 1.0 - percent_over
                accel_factor = max(0.1, accel_factor) # Minimum 10%
                
            self.velocity += heading * ACCELERATION * accel_factor * dt
        elif self.accelerating == -1:
            dot = self.velocity.dot(heading)
            if dot > 10: 
                self.velocity -= heading * BRAKE_STRENGTH * dt
            else: 
                self.velocity -= heading * ACCELERATION * dt 

        # 4. Drag and Friction
        if self.velocity.length() > 0:
            speed_val = self.velocity.length()
            drag_force = -self.velocity.normalize() * (DRAG * speed_val * speed_val)
            friction_force = -self.velocity.normalize() * FRICTION
            
            if speed_val < FRICTION * dt:
                self.velocity = pygame.math.Vector2(0, 0)
            else:
                self.velocity += (drag_force + friction_force) * dt

        # 5. Lateral Traction (Drift)
        if self.velocity.length() > 0:
            forward_velocity = heading * (self.velocity.dot(heading))
            lateral_velocity = self.velocity - forward_velocity
            self.velocity -= lateral_velocity * (DRIFT_FACTOR * dt)

        # 6. Update Position
        self.pos += self.velocity * dt
        self.speed = self.velocity.length()
        
        # Calculate acceleration for display (pixels/sec^2)
        if dt > 0:
            raw_accel = (self.speed - self.last_speed) / dt
            self.instant_acceleration = max(0, raw_accel)
        self.last_speed = self.speed
        
        # Max Speed Cap
        if self.speed > MAX_SPEED:
            self.velocity.scale_to_length(MAX_SPEED)
            self.speed = MAX_SPEED
            
        # 7. Visual Rotation
        self.rotate()
        self.rect.center = (round(self.pos.x), round(self.pos.y))

    def rotate(self):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.mask = pygame.mask.from_surface(self.image)

    def check_collision(self, track_mask):
        self.collided = False
        if track_mask:
            offset = (self.rect.x, self.rect.y)
            if track_mask.overlap(self.mask, offset):
                self.collided = True
                
                # Robust collision: revert position
                self.pos = self.prev_pos.copy()
                self.rect.center = (round(self.pos.x), round(self.pos.y))
                
                # Kill velocity (or bounce slightly)
                # Simple stop is most robust against getting stuck
                self.velocity *= 0.5 
                self.speed = self.velocity.length()
                self.last_speed = self.speed
