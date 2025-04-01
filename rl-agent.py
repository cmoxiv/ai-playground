
import traceback as tb
from ai.rlenv3 import MazeEnv
from ai.rlenv3 import VEGETATION, SWAMP, OBSTACLE, LAVA, EMPTY  


import numpy as np
import gymnasium as gym
from collections import defaultdict

def hash_obs(obs):
    # Convert observation to a hashable type
    # return tuple(obs.astype(np.int32))
    return tuple(obs.flatten().tolist())

class ChatGPT_RLAgent:
    """RL Agent which operates and learns from a gym-compatible
    environment.
    """
    def __init__(self, gamma=0.99, alpha=0.1, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01, action_space_size=5):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        self.n_table = defaultdict(lambda: 0)
        self.prev_state = None
        self.prev_action = None

    def get_action(self, state):
        """Accepts state from the environment and returns an
        action. The agent also learns while getting an action if learn=True."""
        # Learn from the previous state and action before choosing a new one
        state_hashed = hash_obs(state)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(len(self.q_table[state_hashed]))
        else:
            action = np.argmax(self.q_table[state_hashed])

        # Save current state and action for next learning step
        self.prev_state = state
        self.prev_action = action

        return action

    def learn(self, state, reward, terminated):
        state_hashed = hash_obs(state)
        prev_state_hashed = hash_obs(self.prev_state)
        
        
        if self.prev_state is None or self.prev_action is None:
            return
        best_next_action = np.argmax(self.q_table[state_hashed])
        td_target = reward + (0 if terminated else self.gamma * self.q_table[state_hashed][best_next_action])
        # td_target = reward + self.gamma * self.q_table[state_hashed][best_next_action]
        td_error = td_target - self.q_table[prev_state_hashed][self.prev_action]

        self.q_table[prev_state_hashed][self.prev_action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Reset previous state/action if episode ended
        if terminated:
            self.prev_state = None
            self.prev_action = None
            pass
        pass
    pass



class MyAgent():
    def __init__(self, action_space_size,
                 gamma=0.99):
        self.gamma = gamma
        self.nActions = action_space_size
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        self.n_table = defaultdict(lambda: np.ones(action_space_size))
        self.prev_state = None
        self.prev_action = None
        self.epsilon = 1.0
        self.epsilon_decay = .999
        self.epsilon_min = .01

    def _hash(self, state):
        return tuple(state.flatten().tolist())

    def get_action(self, state):
        hashed_state = self._hash(state)
        action = np.argmax(self.q_table[hashed_state])

        if np.random.rand() < self.epsilon:
            action = np.random.choice(range(len(self.q_table[hashed_state])))

        self.prev_state = state
        self.prev_action = action

        return action
    
    def learn(self, state, reward, terminated):
        hashed_state = self._hash(state)

        if self.prev_state is None or self.prev_action is None:
            return
        
        prev_hashed_state = self._hash(self.prev_state)
        best_next_action = np.argmax(self.q_table[hashed_state])

        td_target = reward + (0 if terminated else self.gamma * self.q_table[hashed_state][best_next_action])
        td_error = td_target - self.q_table[prev_hashed_state][self.prev_action]
        
        self.q_table[prev_hashed_state][self.prev_action] += (td_error *
                                                              self.n_table[prev_hashed_state][self.prev_action] / sum(self.n_table[prev_hashed_state]))
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if terminated:
            self.prev_state = None
            self.prev_action = None
        
        return







path_profiles = [  
    (OBSTACLE, OBSTACLE, 0.5, OBSTACLE, .9, 2),
    (VEGETATION, SWAMP, 0.2, OBSTACLE, 0.1, 2),
    (LAVA, OBSTACLE, 0.1, SWAMP, 0.05, 2),
    (SWAMP, VEGETATION, 0.2, OBSTACLE, 0.1, 2)
]

GRID_SIZE = 64
env = MazeEnv(path_profiles=path_profiles, render_mode='human',
              maze_file="./mymaze-dota.txt",
              window_size=512,
              vision_radius=1,
              grid_size=GRID_SIZE)

# env = TimeLimit(env, 200)

state, _ = env.reset()
done = False

agent = MyAgent(action_space_size=env.action_space.n)

import math

episode_reward = 0

try:
    # reward = -0.1
    # terminated = False
    for ep in range(10000):
        for k in range(int(math.sqrt(2*(GRID_SIZE**2)))+10):
            # action = env.action_space.sample()  # Random action
            action = agent.get_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            env.render()
            episode_reward += reward
            
            agent.learn(state, reward, terminated)

            if terminated or truncated:
                # print(f"{done=} {truncated=} {reward=}")
                print(f"{episode_reward=} {agent.epsilon=}")
                episode_reward = 0
                env.reset()
                pass
            pass
except KeyboardInterrupt:
    print("INTERRUPTED BY USER!!")
except BaseException:
    print(tb.format_exc())
finally:
    env.close()
