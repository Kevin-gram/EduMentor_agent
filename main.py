import argparse
import sys
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from stable_baselines3 import DQN  # Import DQN model
from environment.custom_env import EduMentorEnv  # Import custom environment

class EduMentorStaticVisualizer:
    """
    A dynamic visualizer using PyOpenGL to render the EduMentor environment.
    This visualization displays the student, mentors, resources, and state variables.
    """
    def __init__(self, env, model, width=800, height=600):
        self.env = env
        self.model = model
        self.width = width
        self.height = height
        self.current_obs = self.env.reset()  # Initialize the environment
        self.student_position = [self.width // 2, self.height // 2]
        self.student_state = [50.0, 50.0, 50.0]  # Knowledge, Challenge, Interest

        # Simplified mentors and resources for better visualization
        self.mentor_positions = [
            [width // 3, height * 4 // 5],
            [width * 2 // 3, height * 4 // 5]
        ]
        self.resource_positions = [
            [width // 3, height // 5],
            [width * 2 // 3, height // 5]
        ]
        self.mentor_labels = ["Peer", "Expert"]
        self.resource_labels = ["Textbook", "Practice"]

    def init_gl(self):
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)  # Dark background
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluOrtho2D(0, self.width, 0, self.height)

    def draw_circle(self, x, y, radius, r, g, b):
        gl.glColor3f(r, g, b)
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glVertex2f(x, y)
        for angle in range(0, 361, 10):
            rad = np.radians(angle)
            gl.glVertex2f(x + np.cos(rad) * radius, y + np.sin(rad) * radius)
        gl.glEnd()

    def draw_square(self, x, y, size, r, g, b):
        gl.glColor3f(r, g, b)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x - size / 2, y - size / 2)
        gl.glVertex2f(x + size / 2, y - size / 2)
        gl.glVertex2f(x + size / 2, y + size / 2)
        gl.glVertex2f(x - size / 2, y + size / 2)
        gl.glEnd()

    def draw_text(self, x, y, text, r=1, g=1, b=1):
        gl.glColor3f(r, g, b)
        gl.glRasterPos2f(x, y)
        for c in text:
            glut.glutBitmapCharacter(glut.GLUT_BITMAP_8_BY_13, ord(c))

    def draw_grid(self):
        """Draw a grid for better visualization."""
        gl.glColor3f(0.5, 0.5, 0.5)  # Gray grid lines
        for i in range(0, self.width, 100):  # Vertical lines
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(i, 0)
            gl.glVertex2f(i, self.height)
            gl.glEnd()
        for j in range(0, self.height, 100):  # Horizontal lines
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(0, j)
            gl.glVertex2f(self.width, j)
            gl.glEnd()

    def draw_scene(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.draw_grid()  # Draw the grid
        self.draw_text(self.width // 2 - 100, self.height - 30, "EduMentor Environment Visualization")
        student_radius = 20
        self.draw_circle(self.student_position[0], self.student_position[1], student_radius, 0, 0, 0.8)
        self.draw_text(self.student_position[0] - 30, self.student_position[1] - 35, "Student")

        # Draw mentors
        for i, (x, y) in enumerate(self.mentor_positions):
            self.draw_square(x, y, 30, 0.8, 0.2, 0.2)
            self.draw_text(x - 20, y - 25, self.mentor_labels[i])

        # Draw resources
        for i, (x, y) in enumerate(self.resource_positions):
            self.draw_square(x, y, 30, 0.2, 0.7, 0.3)
            self.draw_text(x - 20, y - 25, self.resource_labels[i])

        # Display the current rewards
        reward_text = f"Knowledge: {self.student_state[0]:.1f}, Challenge: {self.student_state[1]:.1f}, Interest: {self.student_state[2]:.1f}"
        self.draw_text(10, self.height - 20, reward_text, r=1, g=1, b=1)  # White text

        gl.glFlush()

    def update(self):
        """Update the agent's position and state variables."""
        # Use the DQN model to predict the next action
        action, _ = self.model.predict(self.current_obs)
        print(f"Predicted action: {action}")  # Debugging log for the predicted action
    
        # Step the environment with the predicted action
        self.current_obs, rewards, done, _ = self.env.step(action)
        print(f"Reward received: {rewards}, Done: {done}")  # Debugging log for rewards and done flag
    
        # Update the agent's position based on the environment's state
        self.student_position[0] = int(self.current_obs[0] * self.width / self.env.grid_size)
        self.student_position[1] = int(self.current_obs[1] * self.height / self.env.grid_size)
        print(f"Updated position: {self.student_position}")  # Debugging log for the agent's position
    
        # Update state variables (knowledge, challenge, interest)
        self.student_state = self.current_obs[2:]  # Update knowledge, challenge, interest
        print(f"Updated state: {self.student_state}")  # Debugging log for the agent's state
    
        # Reset the environment if done
        if done:
            print("Environment reset.")  # Debugging log for environment reset
            self.current_obs = self.env.reset()
    
        # Redraw the scene
        glut.glutPostRedisplay()        
        # Use the DQN model to predict the next action
        action, _ = self.model.predict(self.current_obs)
        self.current_obs, rewards, done, _ = self.env.step(action)

        # Update the agent's position based on the environment's state
        self.student_position[0] = int(self.current_obs[0] * self.width / self.env.grid_size)
        self.student_position[1] = int(self.current_obs[1] * self.height / self.env.grid_size)

        # Update state variables (knowledge, challenge, interest)
        self.student_state = self.current_obs[2:]  # Update knowledge, challenge, interest

        # Reset the environment if done
        if done:
            self.current_obs = self.env.reset()

        # Redraw the scene
        glut.glutPostRedisplay()

    def show(self):
        glut.glutInit(sys.argv)
        glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGB)  # Fixed GLUT_RGB
        glut.glutInitWindowSize(self.width, self.height)
        glut.glutCreateWindow(b"EduMentor Environment Visualization")
        self.init_gl()
        glut.glutDisplayFunc(self.draw_scene)
        glut.glutIdleFunc(self.update)  # Add animation by continuously updating
        glut.glutMainLoop()


def main():
    # Load the trained DQN model
    model_path = "models/dqn/dqn_model"
    env = EduMentorEnv(grid_size=5)
    model = DQN.load(model_path)

    # Initialize the visualizer with the environment and model
    visualizer = EduMentorStaticVisualizer(env, model)
    visualizer.show()


if __name__ == "__main__":
    main()