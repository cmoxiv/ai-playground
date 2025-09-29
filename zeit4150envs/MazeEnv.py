import gymnasium as gym

import numpy as np
import random
import time
import tkinter as tk
from tkinter import filedialog

_root = tk.Tk()
_root.withdraw()   # or use as needed

import pygame
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
    """MazeEnv represents a customizable maze environment for reinforcement learning.

    The maze consists of a grid of tiles, each of which can have
    different properties affecting the agent's health and movement.

    The goal of the agent is to navigate from its starting position to
    the goal tile. Each tile can either heal or harm the agent,
    influencing strategy choices. The maze supports multiple terrain
    profiles, allowing various types of tiles with different behaviors
    and effects.

    The environment is interactive and allows real-time modifications
    to the maze layout via mouse and keyboard controls, enabling quick
    prototyping and experimentation.

    Parameters:
        path_profiles (list, optional): A list of terrain generation profiles.
        vision_radius (int, optional): The radius of tiles visible to the agent.
        render_mode (bool, optional): Whether to render the environment visually.
        window_size (int, optional): The size of the rendering window in pixels.
        grid_size (int, optional): The size of the maze grid.
        episode_length_factor (int, optional): Factor influencing the maximum episode length.
        maze_file (str, optional): File path to a maze configuration to load at initialization.

    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, path_profiles=None,
                 vision_radius=2, render_mode=True,
                 window_size=512, grid_size=64,
                 episode_length_factor=16,
                 maze_file=None):
        """Initialize the environment variables.

        :param path_profiles: List of tuples containing the profile data.
        :param vision_radius: The radius within which the agent can see obstacles or goals.
        :param render_mode: Boolean indicating if rendering is active (True) or not (False).
        :param window_size: The size of the game window in pixels (default: 512).
        :param grid_size: The number of cells per dimension of the maze grid (default: 64).
        :param episode_length_factor: The scaling factor for calculating the maximum number of episodes (default: 16).
        :param maze_file: File path to a custom maze map, or None if using default values.

        """
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

        self.visualise_similar_observations = False
        self.maze_modified = False
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=4, shape=(grid_size, grid_size), dtype=np.int32)

        self.maze = np.full((grid_size, grid_size), OBSTACLE, dtype=np.int32)
        self.maze[self.goal_pos[0], self.goal_pos[1]] = GOAL
        self.loaded_maze = None
        self.maze_file = maze_file or files("zeit4150envs.resources").joinpath("mymaze-dota.txt")
        if self.maze_file:
            self.load_maze_from_file(self.maze_file)
        self.drawing = False
        self.drawn_cells = set()
        self.paused = True
        # self.stored_maze = [False]

        if self.render_mode:
            pygame.init()
            self.font = pygame.font.SysFont('Arial', 16)
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Maze Drawing")
            self.clock = pygame.time.Clock()
            pass
        return

    def clear_maze(self):
        """Clear the maze by setting all tiles to empty."""
        if (self.maze == EMPTY).flat[:-1].all(): 
            self.maze = self.stored_maze.copy()
        else:
            self.stored_maze = self.maze.copy()
            self.maze = np.zeros((self.grid_size, self.grid_size), dtype=int)
            self.maze_modified = True
            self._show_text("Maze Cleared")
        
    def new_maze(self):
        """Initialize a new maze by filling it with obstacles and marks it as modified.

        Args:
            - self: The instance of the class in which this method is being called.
        """
        self.maze.fill(OBSTACLE)
        self.maze_modified = True
        self._show_text("New Maze")
        
    def reset(self, seed=None, pause=False):        
        """Reset the environment to its initial state.

        If the maze has been modified, no action is taken. If a loaded maze exists,
        the current maze is replaced by a copy of the loaded maze and marked as not 
        modified. Otherwise, a new maze filled with obstacles is created and marked as 
        modified. The goal cell is set in the maze, agent position reset to [0, 0], 
        distance from goal calculated based on Hamming distance, initial HP set equal 
        to this distance, current HP set to initial HP, pause state updated based on input,
        episode reward and elapsed steps reset to 0.

        Parameters:
        - seed (int): Seed for the random number generator.
        - pause (bool): Determines if the environment should be paused after resetting.

        Returns:
        - observation (tuple): The observations returned by get_observation() method.
        - info (dict): An empty dictionary as information return, since there is no need to return specific environment-specific information.
        """
        if self.maze_modified:
            pass
        elif self.loaded_maze is not None:
            self.maze = self.loaded_maze.copy()
            self.maze_modified = False
        else:
            self.maze.fill(OBSTACLE)
            self.maze_modified = True
        self.maze[self.goal_pos[0], self.goal_pos[1]] = GOAL
        self.drawn_cells.clear()
        self.agent_pos = [0, 0]
        self.initial_distance = self.hamming_distance_from_goal()
        self.initial_hp = self.initial_distance
        self.hp = float(self.initial_hp)
        self.paused = pause
        self.episode_reward = 0
        self.elapsed_steps = 0
        return self.get_observation(), {}


    def _show_text(self, text):
        """Show the given text on the screen for a specified duration.

        Parameters:
        - text: The message to be displayed.

        Returns:
        None

        Raises:
        No exceptions are raised.
        """
        self.info_text = text
        self.overlay_time = time.time() + 2
        return
        
    def load_maze_from_file(self, filename):
        """Load a maze from the provided filename using numpy's loadtxt function.

        If successful, it sets the loaded maze data to the instance
        variable `self.loaded_maze`, makes a copy of it for
        manipulation as `self.maze`, and then proceeds to mark the
        goal position with a specific value (GOAL). In case an error
        occurs during file loading, it displays an error message.

        Parameters:
        - filename: The name of the file containing maze data in numerical format.

        Returns:
        - None

        Raises:
        - No explicit exceptions are raised, but may catch and handle any general exception that might occur if the file is inaccessible or contains non-integers.
        """
        try:
            self.loaded_maze = np.loadtxt(filename, dtype=np.int32)
            self.maze = self.loaded_maze.copy()
            self._show_text("Maze loaded")
            self.paused = False
            self.maze[self.goal_pos[0], self.goal_pos[1]] = GOAL
        except Exception:
            self._show_text("Failed to load maze")
            pass
        return

    def save_maze_to_file(self, filename):
        """Save the current state of the maze to a specified file.

        Parameters:
        - filename (str): The name of the file where the maze will be saved. The extension should indicate that it is a numpy array (.npy).
        
        Returns:
        - None
        
        Raises:
        - IOError: If there are any issues with writing the maze data to the provided filename.
        - ValueError: If the format specifier '%d' does not meet the expectations for representing maze elements.
        
        Notes: This method utilizes np.savetxt from the numpy library
        to write the maze data to a file in a specified format. Maze
        data must be stored as a 2D array (numpy ndarray) in order for
        this function to work correctly.
        """
        np.savetxt(filename, self.maze, fmt='%d')
        return

    def save_maze(self):
        """Save the current state of the maze as a text file.

        Parameters:
        - filename (str): The name given by the user for saving the maze. This parameter is not directly used in the function but is accessed through tkinter's filedialog.asksaveasfilename() method.

        Returns:
        - None

        Raises:
        - Exception: If there are any issues with the file writing process, an exception will be raised and caught within the try-except block.
        """
        self.paused = True
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.asksaveasfilename(defaultextension=".txt")
        if filename:
            try:
                np.savetxt(filename, self.maze, fmt='%d')
                self._show_text("Maze saved")
            except Exception:
                self._show_text("Failed to save maze")
                pass
            pass
        self.paused = False
        return

    def load_maze(self):
        """Load maze from a text file.

        :param self: Reference to the current object instance.
        :param root: Tkinter root window. 
        :param filedialog: File dialog for selecting a file.
        :return: Returns True if the maze was successfully loaded, False otherwise.
        """
        self.paused = True
        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(defaultextension=".txt")
        if filename:
            self.load_maze_from_file(filename)
            pass
        return

    def hamming_distance_from_goal(self):
        """
        Return the Manhattan distance between two points in a 2D grid.

        Parameters:
            agent_pos: A tuple representing the current position of the agent (x, y).
            goal_pos: A tuple representing the position of the goal (x, y).

        Returns:
            int: The Manhattan distance between the agent's position and the goal's position.
        """
        return abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])

    def update_health(self, tile):
        """
        Update the health of the character based on the tile type.

        Parameters:
        - self: The instance of the character class.
        - tile: The type of tile encountered by the character (VEGETATION or LAVA).

        Returns:
        The updated health of the character after interacting with the given tile.
        """
        self.hp -= self.action_cost
        if tile == VEGETATION:
            self.hp += 3 * self.action_cost
        elif tile == LAVA:
            self.hp -= 0.25 * self.initial_hp
            pass
        return

    def calculate_reward(self, old_pos, new_pos, done):
        """
        Calculate the reward for an agent based on its movement from one position to another in a given environment.

        Parameters:
            old_pos (tuple): The previous position of the agent as a tuple of x and y coordinates.
            new_pos (tuple): The current position of the agent as a tuple of x and y coordinates.
            done (boolean): A flag indicating whether the episode has ended or not.

        Returns:
            float: The calculated reward for the given movement based on the distance to the goal, time penalty, and direction reward. 

        Note: If the agent's position is at the goal, it returns 1000 as a high reward.
        """
        if  self.agent_pos == self.goal_pos:
            return 1000

        old_dist = abs(old_pos[0] - self.goal_pos[0]) + abs(old_pos[1] - self.goal_pos[1])
        new_dist = abs(new_pos[0] - self.goal_pos[0]) + abs(new_pos[1] - self.goal_pos[1])
        direction_reward = self.action_cost if new_dist < old_dist else -self.action_cost
        time_penalty = -self.action_cost
        return direction_reward + time_penalty

    def get_observation_vision(self, pos):
        """
        Get the vision observation at a specific position on the board.

        Parameters:
        - pos (tuple[int, int]): The x and y coordinates representing the position to get the observation for.

        Returns:
        - dict: A dictionary containing the observation data for the specified position. 
            It should include keys such as 'observed_unit', 'observed_type', 'threats', etc.
        """
        ax, ay = pos
        r = self.vision_radius
        obs = np.full((2 * r + 1, 2 * r + 1), OBSTACLE, dtype=np.int32)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = ax + dx, ay + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    obs[dx + r, dy + r] = self.maze[x, y]
        return obs
    
    def get_observation(self):
        """Get observation from a sensor or source.
        
        Returns:
        str: Observation data as string.
        """
        obs = self.get_observation_vision(self.agent_pos)
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
        """Move the robot to the next position in the given direction.
         
        Parameters:
            self (Robot instance): A reference to an instance of Robot class.
            action (str): The direction of movement. Can be "up", "down", "left", or "right".
         
        Returns:
            None
         
        Raises:
            ValueError: If action is not one of the valid directions ("up", "down", "left", or "right").
         
        Note: This method is intended to simulate a robot's movement in response to human instructions.
        """
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
        """
        Sets the user's profile to a specific index.

        Parameters:
            - self: The instance of the class.
            - profile_index (int): The index number corresponding to the desired user profile.

        Returns:
            - None

        Raises:
            - IndexError: If the given profile index is out of range or invalid.
        """
        if 0 <= profile_index < len(self.path_profiles):
            self.current_profile = self.path_profiles[profile_index]
            # self.info_text = f"Selected Profile {profile_index + 1}: {self.current_profile}"
            # self.overlay_time = time.time() + 2
            self._show_text(f"Selected Profile {profile_index + 1}: {self.current_profile}")

    def apply_profile(self):
        """Apply a given profile to the marked grid cells.
         
        Parameters:
        - self: The instance of the class, refers to the current object being processed.
        
        Returns:
        - None
        
        Raises:
        - NotImplementedError: If the method is not implemented in any child classes.
        """
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
                            pass
                        pass
                    pass
                pass
            pass
        return

    def show_keybinding_guide(self):
        """Display a semi-transparent overlay with the keybinding guide."""
        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        font = pygame.font.SysFont('Arial', 20)
        keys = [
            "0-9: Choose path profile to draw with mouse",
            "n: New blank maze",
            "N: New blank maze (paused)",
            "c: Clear maze",
            "C: Clear maze (paused)",
            "r: Reset maze",
            "R: Reset maze (paused)",
            "l: Load maze",
            "L: Load maze (paused)",
            "S: Save Environment",
            "w/a/s/d: Move agent manullay (paused)", 
            "O: Similar Observations (paused)",
            "h: Show/Hide this guide",
            # Add other keybindings here
        ]
        y = 50
        for key in keys:
            text_surface = font.render(key, True, (255, 255, 255))
            overlay.blit(text_surface, (50, y))
            y += 30
        self.window.blit(overlay, (0, 0))
        pygame.display.flip()
        pygame.time.wait(2000)
        
    def handle_keybindings(self, event):
        """Handle keybindings for a specific GUI component.

        Parameters:
            event (Event): The event triggered by user interaction with keyboard.

        Returns:
            None

        Raises:
            ValueError: If an unknown or unhandled keybinding is detected.
        """
        mods = pygame.key.get_mods()
        SHIFT_HELD = mods & pygame.KMOD_SHIFT

        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
        elif event.key == pygame.K_s and SHIFT_HELD:
            self.save_maze()
        elif event.key == pygame.K_o and SHIFT_HELD:
            self.visualise_similar_observations = not self.visualise_similar_observations
        elif event.key == pygame.K_c and SHIFT_HELD:
            self.clear_maze()
            self.paused = True
        elif event.key == pygame.K_r and SHIFT_HELD:
            self.reset(pause=True)
        elif event.key == pygame.K_w:
            self.paused = False
            self._show_text("UP")
            self.step(0)
            self.paused = True
        elif event.key == pygame.K_a:
            self.paused = False
            self._show_text("LEFT")
            self.step(2)
            self.paused = True
        elif event.key == pygame.K_s:
            self.paused = False
            self._show_text("DOWN")
            self.step(1)
            self.paused = True
        elif event.key == pygame.K_d:
            self.paused = False
            self._show_text("RIGHT")
            self.step(3)
            self.paused = True
        elif event.key == pygame.K_n:
            self.new_maze()
        elif event.key == pygame.K_l:
            self.load_maze()
        elif event.key == pygame.K_c :
            self.clear_maze()
        elif event.key == pygame.K_r:
            self.reset(pause=False)
        elif event.key == pygame.K_h:
            self.show_keybinding_guide()
        elif event.key == pygame.K_q:
            self.close()
            # pygame.quit()
            #sys.exit()
        elif event.unicode.isdigit():
            profile_index = int(event.unicode) - 1
            self.set_profile(profile_index)

    def handle_mouse_drawing(self, event):
        """Handle mouse drawing events.

        This method is called when the user interacts with the drawing
        interface using a mouse.  It receives an event object
        containing information about the interaction and processes it
        to update the drawn image.

        Parameters:
        - event: An instance of the Event class containing details about the mouse action (e.g., position, click count).

        Returns:
        None

        Raises: No exceptions are expected to be raised by this method
        under normal circumstances. However, depending on how the
        event data is processed, potential issues may include handling
        incorrect or unexpected input.

        Notes:
        - The method should consider various possible types of mouse events (e.g., click, drag, release) and adapt the drawn image accordingly.
        - Implementations may need to interact with other methods in charge of managing the drawing state or updating graphical displays.
        """
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

    def render_agent_vision(self, pos):
        """
        Render the agent's vision based on its current position.

        Parameters:
        - pos (tuple): The current position of the agent in the game world,
          represented as a tuple (x, y).

        Returns:
        - rendered Vision: A string representation of the agent's vision based on 
          their current location and game state. This could include visual elements like
          walls, objects, other agents, or any other relevant details that should be visible
          to the agent from its current position.

        Raises:
        - ValueError: If pos is not a valid position in the game world.
        """
        vision_surface = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        vr, ax, ay = self.vision_radius, *pos
        pygame.draw.rect(vision_surface, LIGHT_BLUE,
                         ((ax - vr) * self.cell_size, (ay - vr) * self.cell_size,
                          (2 * vr + 1) * self.cell_size, (2 * vr + 1) * self.cell_size))
        pygame.draw.rect(vision_surface, DARKER_BLUE,
                         ((ax - vr) * self.cell_size, (ay - vr) * self.cell_size,
                          (2 * vr + 1) * self.cell_size, (2 * vr + 1) * self.cell_size), 2)
        self.window.blit(vision_surface, (0, 0))

    def highlight_similar_observation(self):
        """Highlight similar observations within a dataset by comparing them using their respective attributes.

        This method compares each observation to all other
        observations in the dataset and highlights those that have
        similar attribute values.  It can be used for anomaly
        detection, data clustering, or exploring relationships between
        variables.

        Parameters:
            - self: An instance of a class that inherits from this method. Typically it will contain necessary data structures such as an observational dataset.

        Returns:
            - A modified version of the input dataset where similar
              observations are highlighted. The method of highlighting
              can vary depending on the specific implementation and
              may include adding a new attribute, altering existing
              attributes, or using visual cues like color changes.
        """
        vision_area = self.get_observation_vision(self.agent_pos)
        
        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        for x in range(self.grid_size - vision_area.shape[0] + 1):
            for y in range(self.grid_size - vision_area.shape[1] + 1):
                # window = self.maze[x:x + vision_area.shape[0],
                #                    y:y + vision_area.shape[1]]
                window = self.get_observation_vision([y, x])
                if np.array_equal(window, vision_area):
                    # print(f"{x=}, {y=}")
                    # print(vision_area)
                    # print(window)
                    # print()
                    self.render_agent_vision((y, x))
                    # rect = pygame.Rect(
                    #     y * (self.window_size // self.grid_size),
                    #     x * (self.window_size // self.grid_size),
                    #     vision_area.shape[1] * (self.window_size // self.grid_size),
                    #     vision_area.shape[0] * (self.window_size // self.grid_size)
                    # )
                    # pygame.draw.rect(overlay, LIGHT_BLUE, rect)
                    # pygame.draw.rect(overlay, DARKER_BLUE, rect, 2)
        self.window.blit(overlay, (0, 0))
        return
        
    def render_text_overlay(self):
        """
        Render a text overlay on the current frame.

        This method draws a given string of text onto the current video frame at a specified position.
        The rendering process includes setting font, color, and size parameters for the overlaid text,
        providing flexibility to customize the output based on user needs. It can be used in various applications
        where annotations or descriptive messages need to be displayed within a video feed.

        Parameters:
        - self: The instance of the class calling this method.

        Returns:
        - None

        Raises:
        - ValueError: If called without providing appropriate parameters.

        Notes:
        - This method may not work with all environments or configurations.
        - Text overlay rendering is platform-dependent and may vary in appearance and behavior
          across different operating systems or graphical user interface (GUI) libraries.
        """
        if time.time() < self.overlay_time:
            overlay_surface = self.font.render(self.overlay_text, True, BLACK, WHITE)
            self.window.blit(overlay_surface, (10, 10))
            pass
        return

    def render_footer(self):
        """
        Render the footer of a web page.

        This method generates the HTML markup for the footer section of a web page,
        including relevant information such as copyright details, links to terms of service,
        privacy policy, and contact information. The footer may also display social media icons or
        links to other pages on the website. This ensures that the user can easily access important
        information about the site without having to navigate through different sections.

        Returns:
            str: A string containing the HTML markup for the footer section.
        """
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
            pass
        return

    def render(self):
        """
        Render the objects into a formatted string representation.

        Returns:
            str: A string containing the rendered representation of the object.
        """
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

        self.render_agent_vision(self.agent_pos)
        if self.visualise_similar_observations:
            self.highlight_similar_observation()
        self.render_footer()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                self.handle_keybindings(event)
            self.handle_mouse_drawing(event)

        pygame.display.flip()
        self.clock.tick(60)
        return
    
    def close(self):
        """Close the connection to the environment.

        This method is used to gracefully shut down a environment
        connection when it is no longer needed.  It ensures that any
        pending operations are completed and resources are released
        properly.  This helps prevent memory leaks and maintains
        system stability.

        Parameters:
        - None

        Returns:
        - None

        """
        if self.render_mode:
            pygame.quit()
            pass
        return
    pass


