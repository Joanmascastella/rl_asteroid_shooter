import numpy as np
import gymnasium as gym
from gymnasium import spaces
from asteroids.main import MainGameLoop
from asteroids.constants import *
import math

class AsteroidShooterEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # how many objects we’ll track at once
        self.MAX_ASTEROIDS = 200
        self.MAX_SHOTS     = 100

        # our game loop instance
        self.game = MainGameLoop()
        self.frame_dt  = 1/30.0
        self._last_score = 0

        # Initial and dynamic telemetry fields
        self.size = self.game.game_size
        self.clock = self.game.clock
        # Score tracking
        self.current_score = self.game.current_score
        self.high_score = self.game.high_score
        # Player telemetry
        self.player_initial_pos = self.game.player_initial_pos
        self.player_current_pos = self.game.player_current_pos
        self.player_rotation = self.game.player_rotation
        self.player_turn_speed = self.game.player_turn_speed
        self.player_shoot_cooldown = self.game.player_shoot_cooldown
        # Asteroid telemetry
        self.number_of_alive_asteroids = self.game.number_of_alive_asteroids
        self.asteroids_current_pos = self.game.asteroids_current_pos
        self.asteroids_current_vel = self.game.asteroids_current_vel
        self.asteroids_current_dist = self.game.asteroids_current_dist
        self.asteroids_current_abs_angle = self.game.asteroids_current_abs_angle
        self.asteroids_current_rel_angle = self.game.asteroids_current_rel_angle
        self.asteroids_path = self.game.asteroids_path 
        # Shot telemetry
        self.shooter_current_pos = self.game.shooter_current_pos
        self.shooter_current_speed = self.game.shooter_current_speed

        # define obeservation space
        # in the observation space we define a dict. 
        # the dict follows a specific pattern: a key then a box which holds the actual datapoints
        # the box can be of many dimensions it mainly varies between how many data points will be produced for that one key
        # for example for player pos we define 2, meaning we will provide two values, being x and y. 
        # in the spaces.box we also have to define the range of the data. where will the data fall 
        # we usually define it as float values between 0.0 and 1.0 throughout as we want the data that is fed to the models to relative in scale
        # this step would be normalization.
        self.observation_space = spaces.Dict({
            "player_pos":           spaces.Box(0.0, 1.0, (2,), dtype=np.float32),
            "player_rot":           spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "player_cd":            spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            # "player_turn_speed":    spaces.Box(0.0, 1.0, (1,), dtype=np.float32),

            "current_score":        spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "high_score":           spaces.Box(0.0, 1.0, (1,), dtype=np.float32),

            "num_asteroids":        spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "asteroids_pos":        spaces.Box(0.0, 1.0, (self.MAX_ASTEROIDS, 2), dtype=np.float32),
            "asteroids_vel":        spaces.Box(-1.0, 1.0, (self.MAX_ASTEROIDS, 2), dtype=np.float32),
            "asteroids_dist":       spaces.Box(0.0, 1.0, (self.MAX_ASTEROIDS,), dtype=np.float32),
            "asteroids_abs_angle":  spaces.Box(0.0, 1.0, (self.MAX_ASTEROIDS,), dtype=np.float32),
            "asteroids_rel_angle":  spaces.Box(0.0, 1.0, (self.MAX_ASTEROIDS,), dtype=np.float32),
            "asteroids_path":       spaces.Box(0.0, 1.0, (self.MAX_ASTEROIDS, 4), dtype=np.float32),

            "shots_pos":            spaces.Box(0.0, 1.0, (self.MAX_SHOTS, 2), dtype=np.float32),
            # "shots_speed":          spaces.Box(0.0, 1.0, (self.MAX_SHOTS,), dtype=np.float32),
        })

        # Here we define a action space. depending on the action chosen a different action occurs
        # there are two types discrete and continuos 
        # discrete which is what we are using below is a finite collection of potential actions 
        # continuos is where the agent can perform actions from a given low and high range. 
        # for example lets say our agent wants to steer a car, we could define the low as -1 and high as 1. 
        # the agent will choose values between those ranges and depending as such the angle of steering will be reflected
        # 0: down, 1: left, 2: right, 3: up, 4: shoot
        self.action_space = spaces.Discrete(5)
        
        # here we define what each action represents. 
        # this way in main game loop we can call this and get the value for the action taken by the agent
        # and pass it to the player update function
        self._action_to_direction = {
            0: "down",   # Move down
            1: "left",   # Move left
            2: "right",  # Move right
            3: "up",     # Move up
            4: "shoot"   # Shoot
        }

    def _get_obs(self):
        obs = {}

        # — Player —
        px, py = self.game.player_current_pos
        obs["player_pos"]        = np.array([px/SCREEN_WIDTH, py/SCREEN_HEIGHT], dtype=np.float32)
        obs["player_rot"]        = np.array([self.game.player_rotation / 360.0], dtype=np.float32)
        obs["player_cd"]         = np.array([self.game.player_shoot_cooldown / PLAYER_SHOOT_COOLDOWN], dtype=np.float32)
        # obs["player_turn_speed"] = np.array([self.game.player_turn_speed / PLAYER_TURN_SPEED], dtype=np.float32)

        # — Score —
        # Here we assume a max‐score scaling of, say, 1000 points
        obs["current_score"] = np.array([self.game.current_score / 1000.0], dtype=np.float32)
        obs["high_score"]    = np.array([self.game.high_score    / 1000.0], dtype=np.float32)

        # — Asteroids — pad/truncate to MAX_ASTEROIDS
        N = self.MAX_ASTEROIDS
        # allocate
        ap = np.zeros((N,2), dtype=np.float32)
        av = np.zeros((N,2), dtype=np.float32)
        ad = np.zeros((N,),   dtype=np.float32)
        aa = np.zeros((N,),   dtype=np.float32)
        ar = np.zeros((N,),   dtype=np.float32)
        apa = np.zeros((N, 4), dtype=np.float32)  # path start and end points

        # fill
        for i, (pos, vel, dist, abs_ang, rel_ang, ast_pos) in enumerate(zip(
                self.asteroids_current_pos,
                self.asteroids_current_vel,
                self.asteroids_current_dist,
                self.asteroids_current_abs_angle,
                self.asteroids_current_rel_angle,
                self.asteroids_path
            )):
            if i >= N: break
            x, y = pos
            ap[i] = [x/SCREEN_WIDTH, y/SCREEN_HEIGHT]
            vx, vy = vel
            # assume max velocity of, e.g., 200 px/sec
            av[i] = [vx/200.0, vy/200.0]
            # normalize distance by screen‐diagonal
            max_d = (SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)**0.5
            ad[i] = dist / max_d
            aa[i] = abs_ang / 360.0
            ar[i] = rel_ang / 360.0
            # path start and end points
            if ast_pos:
                path_start, path_end = ast_pos
                apa[i] = [
                    path_start[0]/SCREEN_WIDTH, path_start[1]/SCREEN_HEIGHT,
                    path_end[0]/SCREEN_WIDTH,   path_end[1]/SCREEN_HEIGHT
                ]

        obs["asteroids_pos"]       = ap
        obs["asteroids_vel"]       = av
        obs["asteroids_dist"]      = ad
        obs["asteroids_abs_angle"] = aa
        obs["asteroids_rel_angle"] = ar
        obs["asteroids_path"]      = apa
        obs["num_asteroids"] = np.array([len(self.asteroids_current_pos)/N], dtype=np.float32)

        # — Shots — pad/truncate to MAX_SHOTS
        M = self.MAX_SHOTS
        sp = np.zeros((M,2), dtype=np.float32)
        ss = np.zeros((M,),   dtype=np.float32)
        for i, (pos, speed) in enumerate(zip(
                self.shooter_current_pos,
                self.shooter_current_speed,
            )):
            if i >= M: break
            x, y = pos
            sp[i] = [x/SCREEN_WIDTH, y/SCREEN_HEIGHT]
            # assume max shot speed e.g. PLAYER_SHOOT_SPEED
            ss[i] = speed / PLAYER_SHOOT_SPEED

        obs["shots_pos"]   = sp
        # obs["shots_speed"] = ss

        return obs

    def reset(self, *, seed=None, options=None):
        # populates all sprite groups, resets score, etc.
        _ = self.game.reset()    
        self._last_score = 0
        # return the first observation
        return self._get_obs(), {}
    

    def step(self, action):
        # define the space in which the agent can move 
        max_d   = (SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)**0.5
        
        # get the current distrubution of asteroids 
        dists   = self.game.asteroids_current_dist
        # calculate the min collection of asteroids in the space
        prev_min = min(dists) if dists else max_d
        # calculate the avg
        prev_avg = sum(dists)/len(dists) if dists else max_d

        # call the main game loop to apply user action
        self.game.apply_action(action)
        # update the game state
        done = self.game.update(self.frame_dt)
        # get the new obs
        obs  = self._get_obs()

        # re intialize the reward
        reward = 0.0    
        # calculate the diff between the current score and the last score achieved 
        delta = self.game.current_score - self._last_score

         # kills + proximity‐weighted bonus
        if delta > 0:
            prox = 1 - (prev_min / max_d)
            reward += 2.0 * delta
            reward += 0.8 * delta * (1 + prox)

        # # ── Action shaping ───
        # if action in [1,2,3]:        # lateral moves
        #     reward += 0.1
        if action == 0:            # back-up is extra valuable
            reward += 0.99
        elif action == 4:            # shoot
            # small cost per shot
            reward -= 0.02
            # penalty if no kill
            if delta == 0:
                reward -= 0.1

        # Bonus for moving into sparse regions
        # if dists:
        #     new_avg   = sum(self.game.asteroids_current_dist)/len(self.game.asteroids_current_dist)
        #     avg_delta = new_avg - prev_avg
        #     reward   += max(-0.05, min(0.05, (avg_delta / max_d) * 0.8))

        # Dodge bonus (getting farther from closest)
        if dists:
            new_min     = min(self.game.asteroids_current_dist)
            dodge_delta = new_min - prev_min
            prox        = 1 - (prev_min / max_d)
            reward     += max(-0.05, min(0.05, (dodge_delta / max_d) * 0.8 * (1 + prox)))

            # Border penalty
            px, py = self.game.player_current_pos
            margin = 250
            if px < margin or px > SCREEN_WIDTH - margin or py < margin or py > SCREEN_HEIGHT - margin:
                reward -= 10
    
            # ── Path danger penalty ──
            danger_threshold = 40  # pixels, adjust as needed
            asteroids_in_path = []
            path_distances = []
            for idx, (path_start, path_end) in enumerate(self.game.asteroids_path):
                x0, y0 = px, py
                x1, y1 = path_start
                x2, y2 = path_end
                dx, dy = x2 - x1, y2 - y1
                if dx == dy == 0:
                    dist = math.hypot(x0 - x1, y0 - y1)
                else:
                    t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
                    proj_x = x1 + t * dx
                    proj_y = y1 + t * dy
                    dist = math.hypot(x0 - proj_x, y0 - proj_y)
                if dist < danger_threshold:
                    reward -= 2  # penalty for being in asteroid path
                    asteroids_in_path.append(idx)
                    path_distances.append(dist)

        # Death penalty
        if done:
            reward -= 100.0

        # ── Path kill bonus ──
        if delta > 0 and asteroids_in_path:
            # Agent killed at least one asteroid and was in the path of one or more
            # Bonus increases with distance from path (up to danger_threshold)
            for dist in path_distances:
                bonus = 10.0 * (dist / danger_threshold)  # farther = bigger bonus
                reward += bonus

        # Wrap up
        # update the last score 
        # return new obs reward and info
        self._last_score = self.game.current_score
        info = {"asteroids_alive": self.game.number_of_alive_asteroids}
        return obs, reward, done, False, info

    def render(self):
        self.game.render()
