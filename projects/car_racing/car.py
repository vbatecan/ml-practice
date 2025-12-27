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
        self.acceleration_val = 0
        self.speed = 0
        
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, dt, track_mask=None):
        self.input()
        self.move(dt)
        self.check_collision(track_mask)

    def input(self):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w]:
            self.acceleration_val = ACCELERATION
        elif keys[pygame.K_s]:
            self.acceleration_val = -ACCELERATION
        else:
            self.acceleration_val = 0

        # Rotate only if moving (arcade feel)
        if abs(self.speed) > 0.1:
            rotation_dir = 1 if self.speed > 0 else -1
            if keys[pygame.K_a]:
                self.angle += ROTATION_SPEED * rotation_dir
            elif keys[pygame.K_d]:
                self.angle -= ROTATION_SPEED * rotation_dir

    def move(self, dt):
        # Apply acceleration
        self.speed += self.acceleration_val
        
        # Friction
        if self.speed > 0:
            self.speed -= FRICTION
            if self.speed < 0: self.speed = 0
        elif self.speed < 0:
            self.speed += FRICTION
            if self.speed > 0: self.speed = 0
            
        # Max speed
        if self.speed > MAX_SPEED:
            self.speed = MAX_SPEED
        if self.speed < -MAX_SPEED / 2:
            self.speed = -MAX_SPEED / 2
            
        # Compute velocity vector
        # Angle 0 is UP (negative Y)
        # Rotation is CCW
        rad = math.radians(self.angle)
        
        dx = -math.sin(rad) * self.speed
        dy = -math.cos(rad) * self.speed
        
        self.pos += pygame.math.Vector2(dx, dy)
        
        # Boundary Check - Removed for large world
        # self.pos.x = max(0, min(self.pos.x, WIDTH))
        # self.pos.y = max(0, min(self.pos.y, HEIGHT))
        
        self.rotate()
        self.rect.center = (round(self.pos.x), round(self.pos.y))

    def rotate(self):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.mask = pygame.mask.from_surface(self.image)

    def check_collision(self, track_mask):
        if track_mask:
            offset = (self.rect.x, self.rect.y)
            # If overlap returns a point, collision occurred
            if track_mask.overlap(self.mask, offset):
                # Simple collision response: bounce back and stop
                self.speed = -self.speed * 0.5
                
                # Move out of collision slightly to prevent sticking
                rad = math.radians(self.angle)
                self.pos -= pygame.math.Vector2(-math.sin(rad), -math.cos(rad)) * 5
                self.rect.center = (round(self.pos.x), round(self.pos.y))
