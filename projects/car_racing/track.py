import pygame
import math
from settings import *
from utils import generate_track_points, compute_track_polygon

class Track(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((WIDTH, HEIGHT))
        self.rect = self.image.get_rect(topleft=(0, 0))
        self.mask = None
        self.start_position = (WIDTH // 2, HEIGHT // 2)
        self.generate()

    def generate(self):
        # 1. Generate path centered at (0,0) roughly
        # We give a large initial radius to ensure points aren't too cramped? 
        # Actually, center=(0,0) works fine, we just need to shift them later.
        raw_center = (0, 0)
        control_points = generate_track_points(raw_center, NUM_POINTS, MIN_RADIUS, MAX_RADIUS)
        raw_path = compute_track_polygon(control_points)
        
        # 2. Compute bounds
        if not raw_path: 
            return # Should not happen

        min_x = min(p[0] for p in raw_path)
        max_x = max(p[0] for p in raw_path)
        min_y = min(p[1] for p in raw_path)
        max_y = max(p[1] for p in raw_path)
        
        # 3. Determine new surface size
        width = int(max_x - min_x + WORLD_PADDING * 2)
        height = int(max_y - min_y + WORLD_PADDING * 2)
        
        self.image = pygame.Surface((width, height))
        self.rect = self.image.get_rect(topleft=(0, 0)) # World coordinates (0,0)? 
        # Actually, let's keep track placed at (0,0) in world space, 
        # but since we shifted points, the "world" effectively starts at 0,0 relative to the image.
        
        # 4. Shift points
        # We want (min_x, min_y) to be at (WORLD_PADDING, WORLD_PADDING)
        offset_x = -min_x + WORLD_PADDING
        offset_y = -min_y + WORLD_PADDING
        
        self.path = [(p[0] + offset_x, p[1] + offset_y) for p in raw_path]
        
        # 5. Draw
        self.image.fill(GRASS_GREEN)
        
        if len(self.path) > 2:
            layers = [
                (WHITE, TRACK_WIDTH + 10),
                (RED, TRACK_WIDTH + 2),
                (GREY, TRACK_WIDTH)
            ]
            
            for color, line_width in layers:
                # Draw the lines
                pygame.draw.lines(self.image, color, True, self.path, width=line_width)
                # Draw rounds at joints
                for p in self.path:
                    pygame.draw.circle(self.image, color, (int(p[0]), int(p[1])), line_width // 2)

            # Draw center line (dashed white)
            dash_length = 20
            gap_length = 30
            for i in range(len(self.path)):
                p1 = pygame.math.Vector2(self.path[i])
                p2 = pygame.math.Vector2(self.path[(i + 1) % len(self.path)])
                
                segment_vec = p2 - p1
                segment_dist = segment_vec.length()
                if segment_dist == 0: continue
                segment_dir = segment_vec.normalize()
                
                curr_dist = 0
                while curr_dist < segment_dist:
                    # Draw a dash
                    start_p = p1 + segment_dir * curr_dist
                    end_dist = min(curr_dist + dash_length, segment_dist)
                    end_p = p1 + segment_dir * end_dist
                    
                    pygame.draw.line(self.image, WHITE, start_p, end_p, 3)
                    curr_dist += dash_length + gap_length

        # 6. Start Position
        if self.path:
            self.start_position = self.path[0]
            
        # 7. Mask (Walls = 1, Track = 0)
        # Re-create mask surface matching new dimensions
        mask_surf = pygame.Surface((width, height))
        mask_surf.fill((255, 255, 255)) # Wall
        if len(self.path) > 2:
            pygame.draw.lines(mask_surf, (0, 0, 0), True, self.path, width=TRACK_WIDTH)
            for p in self.path:
                pygame.draw.circle(mask_surf, (0, 0, 0), (int(p[0]), int(p[1])), TRACK_WIDTH // 2)
            
        self.mask = pygame.mask.from_threshold(mask_surf, (255, 255, 255), threshold=(10, 10, 10, 10))

    def reset(self):
        self.generate()
