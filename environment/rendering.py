import sys
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

class EduMentorVisualizer:
    """
    Visualizer for the EduMentor environment using PyOpenGL.
    Displays a grid-based environment with the agent, mentors, and resources.
    """
    def __init__(self, grid_size=5, cell_size=100):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.agent_pos = [0, 0]
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
    
    def init_gl(self):
        gl.glClearColor(0.9, 0.9, 0.9, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluOrtho2D(0, self.width, 0, self.height)
    
    def draw_grid(self):
        gl.glColor3f(0.8, 0.8, 0.8)
        for i in range(self.grid_size + 1):
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(i * self.cell_size, 0)
            gl.glVertex2f(i * self.cell_size, self.height)
            gl.glVertex2f(0, i * self.cell_size)
            gl.glVertex2f(self.width, i * self.cell_size)
            gl.glEnd()
    
    def draw_agent(self):
        x, y = self.agent_pos
        gl.glColor3f(0.2, 0.6, 0.8)  # Blue color for the agent
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(x * self.cell_size, y * self.cell_size)
        gl.glVertex2f((x + 1) * self.cell_size, y * self.cell_size)
        gl.glVertex2f((x + 1) * self.cell_size, (y + 1) * self.cell_size)
        gl.glVertex2f(x * self.cell_size, (y + 1) * self.cell_size)
        gl.glEnd()
    
    def draw_scene(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.draw_grid()
        self.draw_agent()
        gl.glFlush()
    
    def update_agent_position(self, pos):
        self.agent_pos = pos
        self.draw_scene()
    
    def show(self):
        glut.glutInit(sys.argv)
        glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GL_RGB)
        glut.glutInitWindowSize(self.width, self.height)
        glut.glutCreateWindow(b"EduMentor Environment Visualization")
        self.init_gl()
        glut.glutDisplayFunc(self.draw_scene)
        glut.glutMainLoop()