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
        )  # [x, y, knowledge, challenge, interest]
        self.targets = [
            [0, grid_size - 1],  # Peer
            [grid_size - 1, grid_size - 1],  # Expert
            [0, 0],  # Textbook
            [grid_size - 1, 0],  # Practice
        ]
        self.current_target_index = 0
        self.reset()
    
    def reset(self):
        # Initialize the agent's position and state variables
        self.agent_pos = [0, 0]  # Start at the top-left corner of the grid
        self.state = np.array([0, 0, 50.0, 50.0, 50.0], dtype=np.float32)  # [x, y, knowledge, challenge, interest]
        self.steps = 0
        self.current_target_index = 0  # Start with the first target
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

        # Calculate distance to the current target
        target_pos = self.targets[self.current_target_index]
        distance_to_target = np.linalg.norm(np.array(self.agent_pos) - np.array(target_pos))

        # Reward for reducing distance to the target
        reward += -distance_to_target  # Negative distance as reward (closer = higher reward)

        # Check if the agent has reached the current target
        if self.agent_pos == target_pos:
            reward += 50  # High reward for reaching the target
            self.current_target_index += 1  # Move to the next target
            if self.current_target_index >= len(self.targets):
                self.current_target_index = 0  # Loop back to the first target

        # Clip state values to stay within bounds
        self.state[2:] = np.clip(self.state[2:], 0, 100)

        # Define termination conditions
        done = bool(self.steps >= 50)  # End episode after 50 steps

        return self.state, reward, done, {}
    
    def render(self, mode="human"):
        print(f"Step: {self.steps}, Position: {self.agent_pos}, State: {self.state[2:]}, Target: {self.targets[self.current_target_index]}")
    
    def close(self):
        pass