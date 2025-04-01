import gymnasium as gym
import math
import pygame
import numpy as np
import random
import time
import tkinter as tk
from tkinter import filedialog
from gymnasium import spaces

from importlib.resources import files

# Color constants for rendering
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
DARK_GREEN = (0, 100, 0)
GREENISH_GRAY = (105, 130, 105)
GRAY = (128, 128, 128, 100)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
LIGHT_BLUE = (173, 216, 230, 100)
DARKER_BLUE = (0, 0, 139)

# Tile types
EMPTY = 0
OBSTACLE = 1
LAVA = 2
VEGETATION = 3
SWAMP = 4
GOAL = 5

# HP levels
HP_LEVELS = {
    'underfed': 0,
    'low': 1,
    'medium': 2,
    'high': 3,
    'overfed': 4
}


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, path_profiles=None,
                 vision_radius=2, render_mode=True,
                 window_size=512, grid_size=64,
                 episode_length_factor=16,
                 maze_file=None):
        super().__init__()
        self.path_profiles = path_profiles or [  
            (OBSTACLE, OBSTACLE, 0.5, OBSTACLE, .9, 2),
            (VEGETATION, SWAMP, 0.2, OBSTACLE, 0.1, 2),
            (LAVA, OBSTACLE, 0.1, SWAMP, 0.05, 2),
            (SWAMP, VEGETATION, 0.2, OBSTACLE, 0.1, 2)
        ]
        self.render_mode = render_mode
        self.window_size = window_size
        self.grid_size = grid_size
        self.cell_size = window_size // grid_size
        # self.path_profiles = path_profiles
        self.current_profile = self.path_profiles[0]
        self.overlay_time = 0
        self.overlay_text = ""
        self.vision_radius = vision_radius

        self.agent_pos = [0, 0]
        self.goal_pos = [grid_size - 1, grid_size - 1]

        self.initial_distance = self.hamming_distance_from_goal()
        self._max_episode_steps = episode_length_factor * self.initial_distance
        self.initial_hp = self.initial_distance
        self.hp = self.initial_hp
        self.action_cost = 1 / self.initial_distance if self.initial_distance != 0 else 0
        self.episode_reward = 0
        self.info_text = ""
        self.elapsed_steps = 0

        self.maze_modified = False
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=4, shape=(grid_size, grid_size), dtype=np.int32)

        self.maze = np.full((grid_size, grid_size), OBSTACLE, dtype=np.int32)
        self.maze[self.goal_pos[0], self.goal_pos[1]] = GOAL
        self.loaded_maze = None
        self.maze_file = maze_file or files("zeit4150envs.resources").joinpath("mymaze-dota.txt")
        if self.maze_file:
            self.load_maze_from_file(files.maze_file)
        self.drawing = False
        self.drawn_cells = set()
        self.paused = True

        if self.render_mode:
            pygame.init()
            self.font = pygame.font.SysFont('Arial', 16)
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Maze Drawing")
            self.clock = pygame.time.Clock()
            
    def new_maze(self):
        self.maze.fill(OBSTACLE)
        self._show_text("New Maze")
        
    def reset(self, seed=None, options=None):
        if self.maze_modified:
            pass
        elif self.loaded_maze is not None:
            self.maze = self.loaded_maze.copy()
            self.maze_modified = False
        else:
            self.maze.fill(OBSTACLE)
        self.maze[self.goal_pos[0], self.goal_pos[1]] = GOAL
        self.drawn_cells.clear()
        self.agent_pos = [0, 0]
        self.initial_distance = self.hamming_distance_from_goal()
        self.initial_hp = self.initial_distance
        self.hp = float(self.initial_hp)
        self.paused = False
        self.episode_reward = 0
        self.elapsed_steps = 0
        return self.get_observation(), {}

    def load_maze_from_file(self, filename):
        try:
            self.loaded_maze = np.loadtxt(filename, dtype=np.int32)
            self.maze = self.loaded_maze.copy()
            self._show_text("Maze loaded")
            self.paused = False
            self.maze[self.goal_pos[0], self.goal_pos[1]] = GOAL
        except Exception as e:
            self._show_text("Failed to load maze")

    def _show_text(self, text):
        self.info_text = text
        self.overlay_time = time.time() + 2

    def save_maze(self):
        self.paused = True
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if filename:
            np.savetxt(filename, self.maze, fmt='%d')
            self._show_text("Maze saved")

    def load_maze(self):
        self.paused = True
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(defaultextension=".txt")
        if filename:
            self.load_maze_from_file(filename)

    def hamming_distance_from_goal(self):
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

    def update_health(self, tile):
        self.hp -= self.action_cost
        if tile == VEGETATION:
            self.hp += 3 * self.action_cost
        elif tile == LAVA:
            self.hp -= 0.25 * self.initial_hp

    def calculate_reward(self, old_pos, new_pos, done):
        if  self.agent_pos == self.goal_pos:
            return 1000

        old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])
        new_dist = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
        direction_reward = self.action_cost if new_dist < old_dist else -self.action_cost
        time_penalty = -self.action_cost
        return direction_reward + time_penalty

    def get_observation(self):
        ax, ay = self.agent_pos
        r = self.vision_radius
        obs = np.full((2 * r + 1, 2 * r + 1), OBSTACLE, dtype=np.int32)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = ax + dx, ay + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    obs[dx + r, dy + r] = self.maze[x, y]

        hp_ratio = self.hp / self.initial_hp
        if hp_ratio < 0.1:
            hp_status = HP_LEVELS['underfed']
        elif hp_ratio < 0.4:
            hp_status = HP_LEVELS['low']
        elif hp_ratio < 0.7:
            hp_status = HP_LEVELS['medium']
        elif hp_ratio <= 1.0:
            hp_status = HP_LEVELS['high']
        else:
            hp_status = HP_LEVELS['overfed']

        flat_obs = obs.flatten()
        return np.concatenate([flat_obs, np.array([hp_status], dtype=np.int32)])

    def step(self, action):
        obs = self.get_observation()
        
        self.elapsed_steps += 1
        done = False
        truncated = False

        if self.paused:
            return obs, 0, False, False, {}

        if self.maze[self.agent_pos[0], self.agent_pos[1]] == SWAMP and random.random() < 0.9:
            action = 4

        if self.hp > self.initial_hp and random.random() < 0.8:
            action = 4

        if self.hp < 0.1 * self.initial_hp and random.random() < 0.8:
            action = random.choice([0, 1, 2, 3])

        old_pos = self.agent_pos.copy()
        new_x, new_y = old_pos
        old_x, old_y = old_pos

        if action == 0:
            new_y = max(old_y - 1, 0)
        elif action == 1:
            new_y = min(old_y + 1, self.grid_size - 1)
        elif action == 2:
            new_x = max(old_x - 1, 0)
        elif action == 3:
            new_x = min(old_x + 1, self.grid_size - 1)

        if self.maze[new_x, new_y] != OBSTACLE:
            self.agent_pos = [new_x, new_y]
        else:
            self.agent_pos = [old_x, old_y]

        tile = self.maze[self.agent_pos[0], self.agent_pos[1]]

        done = self.agent_pos == self.goal_pos
        
        self.update_health(tile)
        reward = self.calculate_reward(old_pos, self.agent_pos, done)

        if self.hp <= 0:
            return obs, reward-1000, False, True, {}

        if self.elapsed_steps >= self._max_episode_steps:
            return obs, reward-1000, False, True, {}
    
        
        self.episode_reward += reward
        return obs, reward, done, False, {}


    def set_profile(self, profile_index):
        if 0 <= profile_index < len(self.path_profiles):
            self.current_profile = self.path_profiles[profile_index]
            # self.info_text = f"Selected Profile {profile_index + 1}: {self.current_profile}"
            # self.overlay_time = time.time() + 2
            self._show_text(f"Selected Profile {profile_index + 1}: {self.current_profile}")

    def apply_profile(self):
        primary, secondary, prob_mix, _, _, width = self.current_profile
        half_width = width // 2
        for gx, gy in self.drawn_cells:
            for dx in range(-half_width-1, half_width+2):
                for dy in range(-half_width-1, half_width+2):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        distance = max(abs(dx), abs(dy))
                        if distance == half_width + 1 and self.maze[nx, ny] != EMPTY:
                            self.maze[nx, ny] = primary if np.random.rand() > prob_mix else secondary

    def handle_keybindings(self, event):
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_n:
            self.new_maze()
        elif event.key == pygame.K_s:
            self.save_maze()
        elif event.key == pygame.K_l:
            self.load_maze()
        elif event.key == pygame.K_q:
            pygame.quit()
            exit()
        elif event.unicode.isdigit():
            profile_index = int(event.unicode) - 1
            self.set_profile(profile_index)

    def handle_mouse_drawing(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.drawing = True
            self.drawn_cells.clear()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.drawing = False
            self.apply_profile()
            self.maze_modified = True
        elif event.type == pygame.MOUSEMOTION and self.drawing:
            mx, my = pygame.mouse.get_pos()
            gx, gy = mx // self.cell_size, my // self.cell_size
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                self.maze[gx, gy] = EMPTY
                self.drawn_cells.add((gx, gy))

    def render_agent_vision(self):
        vision_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        vr, ax, ay = self.vision_radius, *self.agent_pos
        pygame.draw.rect(vision_surface, LIGHT_BLUE,
                         ((ax - vr) * self.cell_size, (ay - vr) * self.cell_size,
                          (2 * vr + 1) * self.cell_size, (2 * vr + 1) * self.cell_size))
        pygame.draw.rect(vision_surface, DARKER_BLUE,
                         ((ax - vr) * self.cell_size, (ay - vr) * self.cell_size,
                          (2 * vr + 1) * self.cell_size, (2 * vr + 1) * self.cell_size), 2)
        self.window.blit(vision_surface, (0, 0))

    def render_text_overlay(self):
        if time.time() < self.overlay_time:
            overlay_surface = self.font.render(self.overlay_text, True, BLACK, WHITE)
            self.window.blit(overlay_surface, (10, 10))

    def render_footer(self):
        width = self.window.get_width()
        hp_text = f"HP: {self.hp:.1f}"
        reward_text = f"Reward: {self.episode_reward:.2f}"
        info_text = self.info_text

        hp_surface = self.font.render(hp_text, True, BLACK, WHITE)
        reward_surface = self.font.render(reward_text, True, BLACK, WHITE)

        self.window.blit(hp_surface, (10, 10))
        self.window.blit(reward_surface, ((width - reward_surface.get_width()) // 2, 10))

        if time.time() < self.overlay_time:
            info_surface = self.font.render(info_text, True, BLACK, WHITE)
            self.window.blit(info_surface, (width - info_surface.get_width() - 10, 10))

        
    def render(self):
        if not self.render_mode:
            return

        self.window.fill(WHITE)
        colors = [WHITE, BLACK, RED, DARK_GREEN, GREENISH_GRAY, GREEN]

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(self.window, colors[self.maze[x, y]],
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))

        agent_size = self.cell_size // 2
        offset = (self.cell_size - agent_size) // 2
        pygame.draw.rect(self.window, BLUE,
                         (self.agent_pos[0] * self.cell_size + offset, self.agent_pos[1] * self.cell_size + offset,
                          agent_size, agent_size))
        pygame.draw.rect(self.window, GREEN,
                         (self.goal_pos[0] * self.cell_size, self.goal_pos[1] * self.cell_size, self.cell_size, self.cell_size))

        self.render_agent_vision()
        # self.render_text_overlay()
        self.render_footer()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYUP:
                self.handle_keybindings(event)
            self.handle_mouse_drawing(event)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode:
            pygame.quit()
