import pygame
from asteroids.circleshape import CircleShape
import random
from asteroids.constants import *
class Asteroid(CircleShape):
    def __init__(self, x, y, radius):
        super().__init__(x,y,radius)
        self.spawn_position = pygame.Vector2(x, y)
        self.spawn_velocity = pygame.Vector2(self.velocity)  # store initial velocity
        
    def get_path(self, t):
        # Returns position at time t after spawn
        return self.spawn_position + self.spawn_velocity * t
    
    def draw(self, screen):
        pygame.draw.circle(screen,"white",self.position,self.radius,2)
        
    def update(self,dt):
        self.position += self.velocity * dt
        
    def split(self):
        self.kill()
        
        if self.radius <= ASTEROID_MIN_RADIUS:
            return
        random_angle = random.uniform(20,50)
        a = self.velocity.rotate(random_angle)
        b = self.velocity.rotate(-random_angle)
        
        new_radius = self.radius - ASTEROID_MIN_RADIUS
        asteroid = Asteroid(self.position.x, self.position.y, new_radius)
        asteroid.velocity = a * 1.2
        asteroid = Asteroid(self.position.x, self.position.y, new_radius)
        asteroid.velocity = b * 1.2