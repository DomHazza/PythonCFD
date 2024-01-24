import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Define constants
# ================
LENGTH_X = 10.0
LENGTH_Y = 10.0

DELTA_X = 0.5
DELTA_Y = 0.5
DELTA_T = 0.1

NUM_X = int(LENGTH_X/DELTA_X)
NUM_Y = int(LENGTH_Y/DELTA_Y)

VISCOCITY = 0.0
DENSITY = 1.0


class Node:
    def __init__(self) -> None:
        self.__velocity = np.array([0, 0])
        self.__pressure = 0.0

    
    def __repr__(self) -> str:
        return "V: "+str(self.__velocity)+"     "+"P: "+str(self.__pressure)
    
    
    def __str__(self) -> str:
        return "V: "+str(self.__velocity)+"     "+"P: "+str(self.__pressure)

    
    def set_pressure(self, new_pressure):
        self.__pressure = new_pressure


    def set_velocity(self, new_velocity):
        self.__velocity = new_velocity


    def get_pressure(self):
        return self.__pressure
    

    def get_velocity(self):
        return self.__velocity



grid = np.empty(
    (NUM_X, NUM_Y),
    dtype=Node
)



# Add in boundary conditions
# ==========================

for y_num, row in enumerate(grid):
    for x_num, node in enumerate(row):
        y_len = y_num/DELTA_Y
        x_len = x_num/DELTA_X
        
        grid[y_num, x_num] = Node() 
        grid[y_num, x_num].set_pressure(x_len*y_len)
        grid[y_num, x_num].set_velocity([-y_len, x_len])




# Solve the PDEs
# ==============
        
for t in range(2):
    p_matrix = np.full(
        (NUM_X, NUM_Y),
        0.0
    )
    c=1.0
    for y_num in range(NUM_Y-2):
        for x_num in range(NUM_X-2):
            p = grid[y_num, x_num].get_pressure()
            p_x1 = grid[y_num, x_num+1].get_pressure()
            p_y1 = grid[y_num+1, x_num].get_pressure()
            p_matrix[y_num, x_num] = p - c*(p_x1 - p)*(DELTA_T/DELTA_X) - c*(p_y1 - p)*(DELTA_T/DELTA_Y)
    
    for y_num in range(1, NUM_Y-2):
        for x_num in range(1, NUM_X-2):
            grid[y_num, x_num].set_pressure(p_matrix[y_num, x_num])





























def display_pressure(grid):
    fig = plt.figure(figsize=(11, 7), dpi=100)

    p_matrix = np.full(
        (NUM_X, NUM_Y),
        0
    )
    for y_num, row in enumerate(grid):
        for x_num, node in enumerate(row):
            p_matrix[y_num, x_num] = node.get_pressure()

    plt.contourf(
        np.linspace(0, LENGTH_X, NUM_X),
        np.linspace(0, LENGTH_Y, NUM_Y),
        p_matrix, 
        cmap=cm.inferno
    ) 
    plt.colorbar()
    plt.title('Pressures')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()




def display_velocities(grid):
    fig = plt.figure(figsize=(7, 7), dpi=100)

    vx_matrix = np.full(
        (NUM_X, NUM_Y),
        0
    )
    vy_matrix = np.full(
        (NUM_X, NUM_Y),
        0
    )
    for y_num, row in enumerate(grid):
        for x_num, node in enumerate(row):
            vx_matrix[y_num, x_num] = node.get_velocity()[0]
            vy_matrix[y_num, x_num] = node.get_velocity()[1]

    plt.streamplot(
        np.linspace(0, LENGTH_X, NUM_X),
        np.linspace(0, LENGTH_Y, NUM_Y),
        vx_matrix,
        vy_matrix
    )
    plt.title('Velocities')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()







print(grid)
display_pressure(grid)
#display_velocities(grid)


