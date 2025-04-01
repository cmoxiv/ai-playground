#!/usr/bin/env python

from zeit4150envs.MazeEnv import MazeEnv

env = MazeEnv()
state, _ = env.reset()

try:
    while True:
        action = env.action_space.sample()
        state, reward, *done, info = env.step(action)
        env.render()
        
        if any(done):
            env.reset()

except KeyboardInterrupt:
    print("INTERRUPTED BY USER!!")
finally:
    env.close()
