import matplotlib.pyplot as plt
import numpy as np

# Define constants
# ================
LENGTH_X = 10.0
LENGTH_Y = 10.0

DELTA_X = 0.5
DELTA_Y = 0.5

VISCOCITY = 0.0
DENSITY = 1.0


class Node:
    def __init__(self) -> None:
        self.__velocity = np.array([0, 0])
        self.__pressure = np.array([0, 0])

    
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



grid = np.full(
    (int(LENGTH_X/DELTA_X), int(LENGTH_X/DELTA_X), 1),
    Node()
)

print(grid)



# Add in boundary conditions
# ==========================

def bound_x_bottom(x, node):
    node.set_pressure(np.array([1, 1]))
    node.set_velocity(np.array([1, 0]))


def bound_x_top(x, node):
    node.set_pressure(np.array([1, 1]))
    node.set_velocity(np.array([1, 0]))


def bound_y_left(y, node):
    node.set_pressure(np.array([1, 1]))
    node.set_velocity(np.array([1, 0]))


def bound_y_right(y, node):
    node.set_pressure(np.array([1, 1]))
    node.set_velocity(np.array([1, 0]))


