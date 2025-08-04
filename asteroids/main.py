import sys
import pygame
import math
from asteroids.constants import *
import asteroids.constants as constants
from asteroids.player import Player
from asteroids.asteroid import Asteroid
from asteroids.asteroidfield import AsteroidField
from asteroids.shot import Shot

class MainGameLoop:
    def __init__(self):
        self.dt = None
        # Initial and dynamic telemetry fields
        self.game_size = None
        self.clock = None
        # Score tracking
        self.current_score = 0
        self.high_score = 0
        # Player telemetry
        self.player_initial_pos = None
        self.player_current_pos = None
        self.player_rotation = None
        self.player_turn_speed = None
        self.player_shoot_cooldown = None
        # Asteroid telemetry
        self.number_of_alive_asteroids = 0
        self.asteroids_current_pos = []
        self.asteroids_current_vel = []
        self.asteroids_current_dist = []
        self.asteroids_current_abs_angle = []
        self.asteroids_current_rel_angle = []  # relative to player orientation
        self.asteroids_path = [] 
        # Shot telemetry
        self.shooter_current_pos = []
        self.shooter_current_speed = []
        # Difficulty scaling: percent speed increase per second
        self.asteroid_speed_increase_rate = 0.05
        # Spawn rate decrease (interval reduction) per second
        self.spawn_rate_decrease_rate = 0.05

        self.upteable = None
        self.drawable = None
        self.asteroids = None
        self.shots = None
        self.screen = None
        self.score_font = None
        self.player = None
        self.asteroids = None
        self.shots = None
        self.HIGH_SCORE_FILE = None

    # Reset game and all states
    def reset(self):
        pygame.init()
        pygame.font.init()
        self.current_score = 0
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.game_size = (SCREEN_WIDTH, SCREEN_HEIGHT)
        clock = pygame.time.Clock()
        self.clock = clock

        # ——— Load high score from disk ———
        self.HIGH_SCORE_FILE = "high_score.txt"
        try:
            with open(self.HIGH_SCORE_FILE, "r") as f:
                self.high_score = int(f.read().strip())
        except Exception:
            self.high_score = 0

        # Prepare font for on-screen display
        self.score_font = pygame.font.Font(None, 36)

        # Sprite groups
        self.updateable = pygame.sprite.Group()
        self.drawable = pygame.sprite.Group()
        self.asteroids = pygame.sprite.Group()
        self.shots = pygame.sprite.Group()

        # Container setup for auto-add
        Shot.containers = (self.shots, self.updateable, self.drawable)
        Asteroid.containers = (self.asteroids, self.updateable, self.drawable)
        AsteroidField.containers = self.updateable
        Player.containers = (self.updateable, self.drawable)

        # Create field and player
        AsteroidField()
        self.player = Player(SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
        # Store initial player position & rotation
        self.player_initial_pos = (self.player.position.x, self.player.position.y)
        self.player_rotation = self.player.rotation

        self.dt = 0
        self.update(0.0)
        return

    # Apply actions from agent
    def apply_action(self, action):
        if action == 1:
            self.player.rotate(-self.dt)
        elif action == 2:
            self.player.rotate(self.dt)
        elif action == 3:
            self.player.move(self.dt)
        elif action == 4:
            self.player.shoot()

    # for each frame we reset all telem data 
    # continue rendering the game until user gets killed 
    def update(self, dt):
            self.dt = dt

            # Reset telemetry
            self.asteroids_current_pos.clear()
            self.asteroids_current_vel.clear()
            self.asteroids_current_dist.clear()
            self.asteroids_current_abs_angle.clear()
            self.asteroids_current_rel_angle.clear()
            self.shooter_current_pos.clear()
            self.shooter_current_speed.clear()
            self.asteroids_path.clear()

            self.updateable.update(dt)

            # # Ramping up difficulty 
            # for a in self.asteroids:
            #     a.velocity *= (1 + self.asteroid_speed_increase_rate * dt)
            # constants.ASTEROID_SPAWN_RATE = max(
            #     0.1,
            #     constants.ASTEROID_SPAWN_RATE * (1 - self.spawn_rate_decrease_rate * dt)
            # )
            
            # cleanup shots so if it leaves bounds they get removed 
            for shot in list(self.shots):
                x,y = shot.position
                if x<0 or x>SCREEN_WIDTH or y<0 or y>SCREEN_HEIGHT:
                    shot.kill()

            # cleanup asteroids so if it leaves bounds they get removed 
            for a in list(self.asteroids):
                x,y = a.position
                if x < -a.radius or x > SCREEN_WIDTH+a.radius or y < -a.radius or y > SCREEN_HEIGHT+a.radius:
                    a.kill()
        

            # clamp player to game bounds
            px = max(self.player.radius, min(self.player.position.x, SCREEN_WIDTH - self.player.radius))
            py = max(self.player.radius, min(self.player.position.y, SCREEN_HEIGHT - self.player.radius))
            self.player.position = pygame.Vector2(px, py)

            # telemetry collection
            for a in self.asteroids:
                pos = (a.position.x, a.position.y)
                vel = (a.velocity.x, a.velocity.y)
                dx, dy = pos[0]-px, pos[1]-py
                dist = math.hypot(dx, dy)
                abs_ang = math.degrees(math.atan2(dy, dx)) % 360
                rel_ang = (abs_ang - self.player.rotation) % 360
                self.asteroids_current_pos.append(pos)
                self.asteroids_current_vel.append(vel)
                self.asteroids_current_dist.append(dist)
                self.asteroids_current_abs_angle.append(abs_ang)
                self.asteroids_current_rel_angle.append(rel_ang)
                path_start = tuple(a.spawn_position)
                path_end   = tuple(a.get_path(5.0))
                self.asteroids_path.append((path_start, path_end))
            self.player_current_pos       = (px, py)
            self.player_rotation          = self.player.rotation
            self.player_turn_speed        = PLAYER_TURN_SPEED
            self.player_shoot_cooldown    = self.player.shoot_timer
            
            for shot in self.shots:
                spos = (shot.position.x, shot.position.y)
                speed=math.hypot(shot.velocity.x, shot.velocity.y)
                self.shooter_current_pos.append(spos)
                self.shooter_current_speed.append(speed)
            self.number_of_alive_asteroids = len(self.asteroids)

            # collisions & scoring
            done = False
            for a in list(self.asteroids):
                if a.collides_with(self.player):
                    done = True
                for shot in list(self.shots):
                    if a.collides_with(shot):
                        shot.kill(); a.split()
                        self.current_score += 1
                        if self.current_score > self.high_score:
                            self.high_score = self.current_score
                            with open(self.HIGH_SCORE_FILE, "w") as f:
                                f.write(str(self.high_score))
            return done
    
    # Render game frame 
    def render(self):
        # let Pygame service its window system messages
        pygame.event.pump()

        self.screen.fill("black")
        for spr in self.drawable:
            spr.draw(self.screen)
        s1 = self.score_font.render(f"Score: {self.current_score}", True, pygame.Color('white'))
        s2 = self.score_font.render(f"High : {self.high_score}", True, pygame.Color('white'))
        self.screen.blit(s1, (10,10))
        self.screen.blit(s2, (10, 10 + s1.get_height() + 5))
        pygame.display.flip()
