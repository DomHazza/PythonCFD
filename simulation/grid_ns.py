import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Define constants
# ================
LENGTH_X = 10.0
LENGTH_Y = 10.0

DELTA_X = 0.5
DELTA_Y = 0.5
DELTA_T = 0.001

NUM_X = int(LENGTH_X/DELTA_X)
NUM_Y = int(LENGTH_Y/DELTA_Y)
POSSON_REPETITIONS = 50
TOTAL_REPETITIONS = 500

VISCOCITY = 1.0
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0



# Generate fields
# ===============
p_matrix = np.ones((NUM_X, NUM_Y))
vx_matrix = np.ones((NUM_X, NUM_Y))
vy_matrix = np.ones((NUM_X, NUM_Y))



# Define calculation functions
# ============================
def central_difference_x(matrix):
    diff = np.zeros_like(matrix, dtype=np.float32)
    diff[1:-1, 1:-1] = (
        (matrix[1:-1, 2:] - matrix[1:-1, 0:-2])/(2*DELTA_X)
    )
    return diff


def central_difference_y(matrix):
    diff = np.zeros_like(matrix, dtype=np.float32)
    diff[1:-1, 1:-1] = (
        (matrix[2:, 1:-1] - matrix[0:-2, 1:-1])/(2*DELTA_X)
    )
    return diff


def laplace(matrix):
    diff = np.zeros_like(matrix, dtype=np.float32)
    diff[1:-1, 1:-1] = (
        (matrix[1:-1, 0:-2] + matrix[0:-2, 1:-1]
         - 4 * matrix[1:-1, 1:-1]
         + matrix[1:-1, 2:] + matrix[2:, 1:-1])/DELTA_X**2
    )
    return diff


# Solve the PDEs
# ==============

for i in range(TOTAL_REPETITIONS):
    a_xx = central_difference_x(vx_matrix)
    a_xy = central_difference_y(vx_matrix)
    a_yx = central_difference_x(vy_matrix)
    a_yy = central_difference_y(vy_matrix)
    laplace_vx = laplace(vx_matrix)
    laplace_vy = laplace(vy_matrix)

    # Tentative solve to get momentum equation without pressure gradient
    vx_tent = (
        vx_matrix + DELTA_T * (
            VISCOCITY*laplace_vx - DELTA_T*(
                vx_matrix * a_xx + vy_matrix * a_yy
            )
        )
    )
    vy_tent = (
        vy_matrix + DELTA_T * (
            VISCOCITY*laplace_vy - DELTA_T*(
                vx_matrix * a_xx + vy_matrix * a_yy
            )
        )
    )

    # Force boundary conditions
    vx_tent[0, :] = 0.0
    vx_tent[:, 0] = 0.0
    vx_tent[:, -1] = 0.0
    vx_tent[-1, :] = HORIZONTAL_VELOCITY_TOP
    vy_tent[0, :] = 0.0
    vy_tent[:, 0] = 0.0
    vy_tent[:, -1] = 0.0
    vy_tent[-1, :] = 0.0

    axx_tent = central_difference_x(vx_tent)
    ayy_tent = central_difference_y(vy_tent)

    # Compute a pressure correction by solving the pressure-poisson equation
    rhs = (
        DENSITY/DELTA_T*(
            axx_tent + ayy_tent
        )
    )

    for j in range(POSSON_REPETITIONS):
        p_next = np.zeros_like(p_matrix)
        p_next[1:-1, 1:-1] = 1/4 * (
            p_matrix[1:-1, 0:-2] + p_matrix[0:-2, 1:-1]
            + p_matrix[1:-1, 2:  ] + p_matrix[2:  , 1:-1]
            - DELTA_X**2 * rhs[1:-1, 1:-1]
        )

        # Pressure Boundary Conditions:
        p_next[:, -1] = p_next[:, -2]
        p_next[0,  :] = p_next[1,  :]
        p_next[:,  0] = p_next[:,  1]
        p_next[-1, :] = 0.0

        p_matrix = p_next
    
    dpdx_matrix = central_difference_x(p_matrix)
    dpdy_matrix = central_difference_y(p_matrix)

    # Correct velocities so fluid stays incompressible
    vx_matrix = (
        vx_tent - DELTA_T/DENSITY * dpdx_matrix
    )
    vy_matrix = (
        vy_tent - DELTA_T/DENSITY * dpdy_matrix
    )

    # Enforce boundary conditions for velocity
    vx_matrix[0, :] = 0.0
    vx_matrix[:, 0] = 0.0
    vx_matrix[:, -1] = 0.0
    vx_matrix[-1, :] = HORIZONTAL_VELOCITY_TOP
    vy_matrix[0, :] = 0.0
    vy_matrix[:, 0] = 0.0
    vy_matrix[:, -1] = 0.0
    vy_matrix[-1, :] = 0.0






def display_fields(p_matrix, vx_matrix, vy_matrix):
    fig = plt.figure(figsize=(11, 7), dpi=100)

    plt.contourf(
        np.linspace(0, LENGTH_X, NUM_X),
        np.linspace(0, LENGTH_Y, NUM_Y),
        p_matrix, 
        cmap=cm.Blues,
        levels=50
    ) 
    plt.colorbar()
    plt.streamplot(
        np.linspace(0, LENGTH_X, NUM_X),
        np.linspace(0, LENGTH_Y, NUM_Y),
        vx_matrix,
        vy_matrix
    )
    plt.title('Pressure & velocity')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()







display_fields(p_matrix, vx_matrix, vy_matrix)

