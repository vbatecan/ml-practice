import pygame
import sys
from settings import *
from car import Car
from track import Track

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
        self.car.radars.clear()

    def update(self):
        self.car.update(self.dt, self.track.mask)

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
        track_pos = self.track.rect.topleft - camera_offset
        self.screen.blit(self.track.image, track_pos)
        
        # Draw car with offset
        car_pos = self.car.rect.topleft - camera_offset
        self.screen.blit(self.car.image, car_pos)
        
        # Draw Sensors / Radars
        for start, end, dist in self.car.radars:
            # Apply camera offset to points
            start_off = start - camera_offset
            end_off = end - camera_offset
            
            line_color = GREEN
            if dist < 50: line_color = RED
            elif dist < 100: line_color = (255, 255, 0) # Yellow
            
            pygame.draw.line(self.screen, line_color, start_off, end_off, 1)
            pygame.draw.circle(self.screen, line_color, (int(end_off.x), int(end_off.y)), 3)
        
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
        self.screen.blit(accel_surf, (20, 50))
        
        # Controls info
        controls_surf = self.font.render("WASD to Drive", True, GREY)
        self.screen.blit(controls_surf, (20, 80))
        
        # Sensor Data
        # Sensor Data
        # We can dynamically stringify them
        # If too many, maybe just show first few or compact
        font_height = 20
        start_y = 105
        
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
