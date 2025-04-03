import gym
import numpy as np

class EduMentorEnv(gym.Env):
    """
    Custom Gym environment for the EduMentor project.
    Grid-based environment where the agent moves within a bounded area.
    Observation: [agent_x, agent_y, knowledge, challenge, interest]
    Actions: 4 discrete actions:
        0: Move up
        1: Move down
        2: Move left
        3: Move right
    """
    def __init__(self, grid_size=5):
        super(EduMentorEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = gym.spaces.Discrete(4)  # 4 actions
        self.observation_space = gym.spaces.Box(
            low=0, high=100, shape=(5,), dtype=np.float32
        )  # [agent_x, agent_y, knowledge, challenge, interest]
        self.reset()
    
    def reset(self):
        # Initialize the agent's position and state variables
        self.agent_pos = [0, 0]  # Start at the top-left corner of the grid
        self.state = np.array([0, 0, 50.0, 50.0, 50.0], dtype=np.float32)  # [x, y, knowledge, challenge, interest]
        self.steps = 0
        return self.state
    
    def step(self, action):
        self.steps += 1
        reward = 0

        # Move the agent within the grid boundaries
        if action == 0 and self.agent_pos[1] < self.grid_size - 1:  # Move up
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0:  # Move down
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0:  # Move left
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:  # Move right
            self.agent_pos[0] += 1

        # Update the state variables
        self.state[:2] = self.agent_pos
        if action == 0:  # Move up (interact with mentor)
            self.state[2] += 10  # Increase knowledge
            self.state[3] -= 5   # Decrease challenge
            self.state[4] += 5   # Increase interest
            reward = 10
        elif action == 1:  # Move down (use resource)
            self.state[2] += 8   # Increase knowledge
            self.state[3] -= 3   # Decrease challenge
            self.state[4] += 3   # Increase interest
            reward = 8
        elif action == 2:  # Move left (take a break)
            self.state[4] += 10  # Increase interest
            reward = 5
        elif action == 3:  # Move right (attempt challenge)
            if self.state[3] > 30:  # Only attempt if challenge is high
                self.state[2] += 15  # Increase knowledge significantly
                self.state[3] -= 10  # Decrease challenge
                reward = 15
            else:
                reward = -5  # Penalize if challenge is too low

        # Clip state values to stay within bounds
        self.state[2:] = np.clip(self.state[2:], 0, 100)

        # Define termination conditions
        done = bool(self.state[2] >= 90 or self.steps >= 50)

        return self.state, reward, done, {}
    
    def render(self, mode="human"):
        print(f"Step: {self.steps}, Position: {self.agent_pos}, State: {self.state[2:]}")
    
    def close(self):
        pass