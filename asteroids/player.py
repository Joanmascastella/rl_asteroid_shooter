import pygame
from asteroids.constants import *
from asteroids.circleshape import CircleShape
from asteroids.shot import Shot
class Player(CircleShape):
    def __init__(self, x, y):
        super().__init__(x,y,PLAYER_RADIUS)
        self.rotation = 0
        self.shoot_timer = 0
    def draw(self, screen):
        pygame.draw.polygon(screen,"white",self.triangle(),2)

    def triangle(self):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        right = pygame.Vector2(0, 1).rotate(self.rotation + 90) * self.radius / 1.5
        a = self.position + forward * self.radius
        b = self.position - forward * self.radius - right
        c = self.position - forward * self.radius + right
        return [a, b, c]
    
    def rotate(self,dt):
        self.rotation += PLAYER_TURN_SPEED * dt
        
    def update(self, dt, action=None):
            self.shoot_timer -= dt
            if action == 1:
                self.move(-dt)
            elif action == 2:
                self.move(dt)
            elif action == 3:
                self.rotate(-dt)
            elif action == 4:
                self.rotate(dt)
            elif action == 5:
                self.shoot()

            
    def shoot(self):
        if self.shoot_timer > 0:
            return
        self.shoot_timer = PLAYER_SHOOT_COOLDOWN
        shot = Shot(self.position.x, self.position.y)
        shot.velocity = pygame.Vector2(0,1).rotate(self.rotation) * PLAYER_SHOOT_SPEED
            
    def move(self,dt):
        forward = pygame.Vector2(0, 1).rotate(self.rotation)
        self.position += forward * PLAYER_SPEED * dt