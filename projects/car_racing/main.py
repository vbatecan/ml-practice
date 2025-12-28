import pygame
import sys
import math
from settings import *
from car import Car
from track import Track
import csv
import datetime
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import get_closest_point_on_path, calculate_cross_track_error, calculate_heading_error

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        
        self.track = Track()
        self.car = Car(self.track.start_position)
        
        self.running = True
        self.telemetry_data = []

        self.total_distance = 0.0
        self.collision_count = 0

        self.model = None
        self.scaler = None
        
        self.current_controls = (0, 0, 0, 0, 0) # w, a, s, d, handbrake
        self.steering_ema = 0.0 # Smoothed steering value (-1 to 1)

        # New Telemetry Vars
        self.cte = 0.0
        self.heading_error = 0.0

        # Flashlight Surface
        if FLASHLIGHT_MODE:
            self.flashlight_surf = pygame.Surface((WIDTH, HEIGHT))
            self.flashlight_surf.fill(BLACK)
            self.flashlight_surf.set_colorkey(WHITE) # White becomes transparent
            # Draw the transparent hole
            pygame.draw.circle(self.flashlight_surf, WHITE, (WIDTH // 2, HEIGHT // 2), FLASHLIGHT_RADIUS)

        if MODEL_PLAYING:
            print("Loading Model...")
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("Model loaded.")
                
                print("Loading Data for Scaler...")
                
                df = pd.read_csv(DATA_PATH)
                x_train_source = df.drop(columns=["W", "A", "S", "D"], axis=1)
                
                self.feature_columns = x_train_source.columns.tolist()

                self.scaler = StandardScaler()
                self.scaler.fit(x_train_source)
                print("Scaler fitted.")
                
                # Pre-calculate fallback radar data
                self.fallback_radar_data = [0] * RADAR_COUNT
                
            except Exception as e:
                print(f"Failed to load model or scaler: {e}")
                print("Falling back to manual control.")
                self.model = None



    def run(self):
        while self.running:
            # Event handling
            self.events()
            
            # Update
            self.dt = self.clock.tick(FPS) / 1000.0
            self.update()
            
            # Draw
            self.draw()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
                if event.key == pygame.K_u:
                    self.save_telemetry()
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def reset_game(self):
        self.track.reset()
        self.car.pos = pygame.math.Vector2(self.track.start_position)
        self.car.velocity = pygame.math.Vector2(0, 0)
        self.car.speed = 0
        self.car.angle = 0
        self.car.last_speed = 0
        self.car.instant_acceleration = 0

        self.car.instant_acceleration = 0
        self.car.radars.clear()
        self.car.instant_acceleration = 0
        self.car.radars.clear()
        
        self.total_distance = 0.0
        self.collision_count = 0


    def save_telemetry(self):
        if not self.telemetry_data:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"racing_data_{timestamp}.csv"
        
        print(f"Saving {len(self.telemetry_data)} rows to {filename}...")
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Create Header
                # W, A, S, D, Handbrake, Speed, SteerAngle, Collision, Acceleration, CTE, HeadingError, R0, R1, ..., Rn
                header = ["W", "A", "S", "D", "Handbrake", "Speed", "SteeringAngle", "Collision", "Acceleration", "CTE", "HeadingError"]
                for i in range(RADAR_COUNT):
                    header.append(f"R{i}")
                
                writer.writerow(header)
                writer.writerows(self.telemetry_data)
            print("Save completed.")
            self.telemetry_data = [] # Clear data after saving
        except Exception as e:
            print(f"Error saving data: {e}")

    def record_telemetry(self):
        # Use current applied controls (whether manual or model)
        if len(self.current_controls) == 5:
            w, a, s, d, hb = self.current_controls
        else:
            w, a, s, d = self.current_controls
            hb = 0
            
        speed = round(self.car.speed, 2)
        angle = round(self.car.angle, 2)
        collided = 1 if self.car.collided else 0
        accel = round(self.car.instant_acceleration, 2)
        
        # Calculate CTE and Heading (re-calculate or use stored)
        # We calculate it here for recording to accuracy
        closest_idx, dist = get_closest_point_on_path(self.car.pos, self.track.path)
        cte = calculate_cross_track_error(self.car.pos, self.track.path, closest_idx)
        heading_err = calculate_heading_error(self.car.angle, self.car.pos, self.track.path, closest_idx, lookahead=15)
        
        self.cte = cte
        self.heading_error = heading_err
        
        # Normalize/Round for CSV
        cte = round(cte, 2)
        heading_err = round(heading_err, 2)
        
        radars = []
        if len(self.car.radars) == RADAR_COUNT:
            for r in self.car.radars:
                dist = int(r[2])
                radars.append(dist)
        else:
            # Fallback if radars not ready
            # Assuming 0 distance (collision/error) -> High danger
            for _ in range(RADAR_COUNT):
                radars.append(0) # Distance
            
            
        row = [w, a, s, d, hb, speed, angle, collided, accel, cte, heading_err] + radars
        
        # De-duplication: Only save if changed (ignore first row for simplicity or check)
        if self.telemetry_data:
            last_row = self.telemetry_data[-1]
            if row == last_row:
                return # Skip duplicate
                
        self.telemetry_data.append(row)


    def update(self):
        controls = None
        
        if MODEL_PLAYING and self.model and self.scaler:
            speed = self.car.speed
            angle = self.car.angle
            collided = 1 if self.car.collided else 0
            accel = self.car.instant_acceleration
            
            # Calculate CTE and Heading for Model
            closest_idx, _ = get_closest_point_on_path(self.car.pos, self.track.path)
            cte = calculate_cross_track_error(self.car.pos, self.track.path, closest_idx)
            heading_err = calculate_heading_error(self.car.angle, self.car.pos, self.track.path, closest_idx, lookahead=15)
            
            self.cte = cte
            self.heading_error = heading_err

            radars = []
            if len(self.car.radars) == RADAR_COUNT:
                for r in self.car.radars:
                    dist = r[2]
                    # Append dist
                    radars.append(dist)
            else:
                radars = self.fallback_radar_data
                
            # Input structure must match training data:
            # Speed, SteeringAngle, Collision, Acceleration, CTE, HeadingError, Radars...
            input_data = [speed, angle, collided, accel, cte, heading_err] + radars
            
            # Create DataFrame with proper column names to avoid warnings
            input_df = pd.DataFrame([input_data], columns=self.feature_columns)
            
            input_scaled = self.scaler.transform(input_df)
            
            prediction = self.model.predict(input_scaled, verbose=0)
            pred = prediction[0] # First sample
            
            # Thresholding for Throttle (W/S) - Keep binary or maybe probability?
            # Usually throttle is fine as on/off, but steering needs smoothing.
            w = 1 if pred[0] > 0.5 else 0
            s = 1 if pred[2] > 0.5 else 0
            
            # Smooth Steering (A/D)
            # Pass raw probability or target to Car, Car handles smoothing.
            raw_a = pred[1]
            raw_d = pred[3]
            
            # Map back to A/D for controls tuple
            # If Raw A > Raw D, we lean left.
            # We can just pass the raw values or a net value.
            # Car.input accepts (w, a, s, d).
            # If we pass (w, raw_a, s, raw_d), Car.input calculates target = raw_a - raw_d.
            # This works perfectly.
            a = 1 if raw_a > 0.5 else 0
            d = 1 if raw_d > 0.5 else 0
            
            # Input Interpretation
            if AI_FLOAT_CONTROL:
                # Use raw probability/strength (0.0 to 1.0)
                w = float(pred[0])
                a = float(pred[1])
                s = float(pred[2])
                d = float(pred[3])
                
                # Check for Handbrake (optional 5th output)
                hb = 0.0
                if len(pred) > 4:
                    hb = float(pred[4])
                
            else:
                # Binary Thresholding
                w = 1 if pred[0] > 0.5 else 0
                s = 1 if pred[2] > 0.5 else 0
                a = 1 if pred[1] > 0.5 else 0
                d = 1 if pred[3] > 0.5 else 0
                
                hb = 0
                if len(pred) > 4:
                     hb = 1 if pred[4] > 0.5 else 0
            
            # Auto-push logic (keep it for now to prevent getting stuck at start)
            if speed < 50:
                w = 1.0
                hb = 0 # Don't handbrake if trying to auto-push

            controls = (w, a, s, d, hb)
            
        # Capture manual controls if model isn't playing
        if controls is None:
            keys = pygame.key.get_pressed()
            w = 1 if keys[pygame.K_w] else 0
            a = 1 if keys[pygame.K_a] else 0
            s = 1 if keys[pygame.K_s] else 0
            d = 1 if keys[pygame.K_d] else 0
            hb = 1 if keys[pygame.K_SPACE] else 0
            
            # Mutual Exclusion for Manual Input
            if w and s:
                w = 0
                s = 0
            if a and d:
                a = 0
                d = 0
            
            controls = (w, a, s, d, hb)
            
        self.current_controls = controls

        self.car.update(self.dt, self.track.mask, controls)
        self.record_telemetry()
        
        # Stats
        if self.car.speed > 0:
            self.total_distance += self.car.speed * self.dt
        if self.car.collided:
            self.collision_count += 1

        
        # Skid marks
        # Calculate lateral velocity to see if drifting
        if self.car.velocity.length() > 50:
            rad = math.radians(self.car.angle)
            heading = pygame.math.Vector2(-math.sin(rad), -math.cos(rad))
            forward_velocity = heading * (self.car.velocity.dot(heading))
            lateral_velocity = self.car.velocity - forward_velocity
            
            if lateral_velocity.length() > 20: # Drifting threshold
                # Draw skid marks on the track image (persistent)
                # Offset for tires (roughly)
                p = self.car.pos
                # We can draw a simple circle or line at tire locations
                # Just draw one faint dark trace for now at center
                pygame.draw.circle(self.track.image, (20, 20, 20), (int(p.x), int(p.y)), 10)


    def draw(self):
        self.screen.fill(BLACK)
        
        # Camera Offset: Center the car on the screen
        # Camera pos = Car pos - Screen Center
        camera_x = self.car.rect.centerx - WIDTH // 2
        camera_y = self.car.rect.centery - HEIGHT // 2
        
        # Offset vector
        camera_offset = pygame.math.Vector2(camera_x, camera_y)
        
        # Draw track with offset
        # self.track.image is the big world surface
        if not RADAR_VIEW_ONLY:
            track_pos = self.track.rect.topleft - camera_offset
            self.screen.blit(self.track.image, track_pos)
        
        # Draw car with offset
        car_pos = self.car.rect.topleft - camera_offset
        self.screen.blit(self.car.image, car_pos)
        
        # Draw Sensors / Radars
        if DRAW_RADARS or RADAR_VIEW_ONLY:
            for start, end, dist in self.car.radars:
                # Apply camera offset to points
                start_off = start - camera_offset
                end_off = end - camera_offset
                
                line_color = GREEN
                if dist < RADAR_DANGER_DIST: line_color = RED
                elif dist < RADAR_WARNING_DIST: line_color = (255, 255, 0) # Yellow
                
                pygame.draw.line(self.screen, line_color, start_off, end_off, 1)
                pygame.draw.circle(self.screen, line_color, (int(end_off.x), int(end_off.y)), 3)
        
        if FLASHLIGHT_MODE:
            self.screen.blit(self.flashlight_surf, (0, 0))
            
        if REALISTIC_VISION:
            # 1. Calculate polygon
            vision_poly = self.car.get_vision_polygon(self.track.mask)
            
            if vision_poly:
                # 2. Create local polygon by applying camera offset
                local_poly = [p - camera_offset for p in vision_poly]
                
                # 3. Reuse fog surface
                if not hasattr(self, 'vision_surf'):
                     self.vision_surf = pygame.Surface((WIDTH, HEIGHT))
                     self.vision_surf.set_colorkey(WHITE)
                
                self.vision_surf.fill(BLACK)
                # self.vision_surf.set_colorkey(WHITE) # Already set
                
                # 4. Draw visibility polygon in WHITE (which becomes transparent)
                if len(local_poly) > 2:
                    pygame.draw.polygon(self.vision_surf, WHITE, local_poly)
                    
                self.screen.blit(self.vision_surf, (0, 0))

        # HUD (Static, no offset)
        self.draw_hud()
        
        pygame.display.flip()

    def draw_hud(self):
        # Semi-transparent panel
        # Make it tally dynamically or just large enough
        hud_surf = pygame.Surface((280, 300), pygame.SRCALPHA)
        pygame.draw.rect(hud_surf, (0, 0, 0, 128), hud_surf.get_rect(), border_radius=10)
        self.screen.blit(hud_surf, (10, 10))

        # Speed
        # Max Phys Speed = 800. Let's call that 200 km/h. Factor = 0.25
        display_speed = int(self.car.speed * 0.25)
        speed_color = GREEN if display_speed < 100 else (RED if display_speed > 160 else WHITE)
        speed_surf = self.font.render(f"Speed: {display_speed} km/h", True, speed_color)
        self.screen.blit(speed_surf, (20, 20))

        # Acceleration Check
        accel_val = int(self.car.instant_acceleration)
        accel_surf = self.font.render(f"Accel: {accel_val}", True, WHITE)
        accel_surf = self.font.render(f"Accel: {accel_val}", True, WHITE)
        self.screen.blit(accel_surf, (20, 50))
        
        # Track Info
        cte_surf = self.font.render(f"CTE: {int(self.cte)}", True, WHITE)
        self.screen.blit(cte_surf, (150, 20))
        
        head_surf = self.font.render(f"Head: {int(self.heading_error)}", True, WHITE)
        self.screen.blit(head_surf, (150, 50))
        
        # Controls info
        controls_surf = self.font.render("AI Inputs:", True, GREY)
        self.screen.blit(controls_surf, (20, 80))
        
        # Draw WASD Visualization
        # Layout:   W
        #         A S D
        #           ^ (Space/HB)
        base_x, base_y = 150, 80
        size = 20
        gap = 5
        
        # Handle 5th element safely
        if len(self.current_controls) == 5:
            cw, ca, cs, cd, chb = self.current_controls
        else:
             cw, ca, cs, cd = self.current_controls
             chb = 0

        
        keys = [
            ("W", float(cw), (base_x + size + gap, base_y)),
            ("A", float(ca), (base_x, base_y + size + gap)),
            ("S", float(cs), (base_x + size + gap, base_y + size + gap)),
            ("D", float(cd), (base_x + (size + gap)*2, base_y + size + gap))
        ]
        
        # Draw WASD
        for k_char, active, (kx, ky) in keys:
            color = GREEN if active > 0.1 else DARK_GREY
            pygame.draw.rect(self.screen, color, (kx, ky, size, size), border_radius=3)
            # Text
            kt = self.font.render(k_char, True, BLACK if active > 0.1 else WHITE)
            kt_rect = kt.get_rect(center=(kx + size//2, ky + size//2))
            self.screen.blit(kt, kt_rect)
            
        # Draw Handbrake (Space) - Wide bar below
        hb_active = float(chb)
        hb_color = RED if hb_active > 0.1 else DARK_GREY
        hb_x = base_x
        hb_y = base_y + (size + gap) * 2
        hb_w = (size + gap) * 3 - gap # Span across A S D
        
        pygame.draw.rect(self.screen, hb_color, (hb_x, hb_y, hb_w, size), border_radius=3)
        hb_text = self.font.render("Space", True, BLACK if hb_active > 0.1 else WHITE)
        hb_rect = hb_text.get_rect(center=(hb_x + hb_w//2, hb_y + size//2))
        self.screen.blit(hb_text, hb_rect)

        
        # Odometer & Collisions
        dist_km = int(self.total_distance / 1000) # Pseudo-km (pixels)
        stats_surf = self.font.render(f"Dist: {dist_km} | Hits: {self.collision_count}", True, WHITE)
        self.screen.blit(stats_surf, (20, 130)) # Moved down slightly
        
        # Sensor Data
        # Sensor Data
        # We can dynamically stringify them
        # If too many, maybe just show first few or compact
        font_height = 20
        start_y = 135

        
        if self.car.radars:
            # Create a string representation
            # e.g. "S0: 100 S1: 200 ..."
            sensor_str = ""
            for i, r in enumerate(self.car.radars):
                dist = int(r[2])
                sensor_str += f"{dist} "
                if len(sensor_str) > 30: # Wrap line
                    sensor_surf = self.font.render(sensor_str, True, WHITE)
                    self.screen.blit(sensor_surf, (20, start_y))
                    start_y += font_height
                    sensor_str = ""
            
            if sensor_str:
                sensor_surf = self.font.render(sensor_str, True, WHITE)
                self.screen.blit(sensor_surf, (20, start_y))
                start_y += font_height

        regen_surf = self.font.render("R: New Track | Esc: Quit", True, GREY)
        self.screen.blit(regen_surf, (20, start_y + 5))



if __name__ == '__main__':
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()
