"""
Implementation of the Potential Field 

I would like to thank Professor Morteza Lahijanian and the fellow course mate Kedar More for 
discussion during the Implementation of this code 

"""

from timeit import default_timer as timer
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import polytope as pc
from workspaces import config

WORKSPACE_CONFIG = config()


# Global Flags
XLIMIN = -10
XLIMAX = 40
NUMS = 200

CENTROID_GAIN = [40, 60, 20, 300 ,60, 50, 50, 50, 50, 50]
OBS_RADIUS =    [5, 5, 7.5, 8 ,3.5, 5.5, 5.5, 5.5, 5.5, 5.5]
                                                             

def attractivePotential(state, ATTRACT_GAIN, DSTARGOAL, q_goal):
    """
    Calculates the Attraction Potential based on the hyper parameters 
    for each state
    PARAMETERS
    ----------
    state : numpy array
    ATTRACT_GAIN : Attraction Gain 
    DSTARGOAL : Attraction distance over which the function is non-linear
    q_goal : Goal 
    """

    distToGoal = np.hypot(state[0] - q_goal[0], state[1] - q_goal[1])

    # unroll for speed
    gain = ATTRACT_GAIN
    dStarGoal = DSTARGOAL

    if distToGoal <= dStarGoal:
        # Linear 
        U_att = 0.5 * gain * distToGoal ** 2
    else:
        # Non - Linear
        U_att = dStarGoal * gain * distToGoal - 0.5 * gain * dStarGoal ** 2

    return U_att

def repulsivePotential(state, obs, REPULSIVE_GAIN, Q_STAR):
    """
    Repulsive Function 
    PARAMETERS
    ----------
    state : Current state of the point robot 
    obs : List of Shapely.Geometry.MultiPolygon 
    REPULSIVE_GAIN : List of Gain for each obstacle 
    Q_STAR : List of repulsive distance gain, from obstacle 
    """
    obstacles = obs
    GAIN = REPULSIVE_GAIN

    # To prevent shattering, sum of all obstacles
    U_rep = 0

    for (obstacle, qStar, gn) in zip(obstacles, Q_STAR, GAIN):
        # Distance to the obstacle 
        distToObst = abs(obstacle.exterior.distance(Point(state)))

        # return NAN in case of collision with the obstacle
        if distToObst == 0:
            distToObst = np.nan
            U_rep = 10

        elif Point(state).within(obstacle):
            # Inside Obstacle
            distToObst = np.nan
            U_rep = 10
        else:
            # Inside Q_Star
            if distToObst <= qStar:
                U_rep += 0.5 * gn * (1 / distToObst - 1 / qStar) ** 2
            else:
                U_rep += 0


    return U_rep

def repulsivePotential2(state, obs):
    """
    PARAMETERS
    ----------
    state : Current state of the point of the robot 
    obs : List of Shapely.Geometry.MultiPolygon 
    """
    global OBS_RADIUS
    global CENTROID_GAIN
    obstacles = obs
    RAD = OBS_RADIUS
    GAIN = CENTROID_GAIN
    dist = []
    U_rep = 0

    for (obstacle, gn, r) in zip(obstacles, GAIN, RAD):
        # return NAN, when on obstacle boundary
        distToObst = abs(obstacle.exterior.distance(Point(state)))
        if distToObst == 0:
            distToObst = np.nan
            U_rep = 10

        elif Point(state).within(obstacle):
            # If inside obstacle
            distToObst = np.nan
            U_rep = 10
        else:
            # Center of Mass/Gravity based 
            cg = obstacle.centroid
            p = Point(state)

            # Elliptical Distance
            dist = np.hypot(p.x - cg.x, p.y - cg.y)
            if dist < r:
                U_rep = gn/(dist**2)
                return U_rep
            else:
                U_rep = 0

    return U_rep    

def potential(state, obs, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR, q_goal):
    """
    Total Potential for a state 
    """
    return (attractivePotential(state, ATTRACT_GAIN, DSTARGOAL, q_goal) +\
         repulsivePotential(state, obs, REPULSIVE_GAIN, Q_STAR))

def potential_large_workspace(state, obs, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR):
    """
    Total Potential for a state including elliptical distance
    """
    return (attractivePotential(state, ATTRACT_GAIN, DSTARGOAL, q_goal) +\
         repulsivePotential(state, obs, REPULSIVE_GAIN, Q_STAR) +\
              repulsivePotential2(state, obs))

def isCloseTo(state, q_goal, epsilon=0.25):
    """
    Returns true with within goal boundary
    PARAMETERS
    ----------
    state: numpy array - current state of the point robot
    q_goal: Goal 
    
    """

    dist = np.linalg.norm(state-q_goal)

    return (dist <= epsilon)

def isAtGoal(state, q_goal):
    """
    Helper Function to check if the current state of the robot is at Goal 
    """

    closeToGoal = isCloseTo(state, q_goal, epsilon=0.25)

    return closeToGoal


def calc_potential_field(q_start, q_goal, obs, NUMS, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR):
    """
    Calculates the potential of the all the states in the grid
    PARAMETERS
    ---------- 
    q_start : Start position of the point robot
    q_goal : Goal Position 
    obs : List of Shapely.Geometry.MultiPolygon 
    NUMS: Grid Size 
    DSTARGOAL : Hyper parameter
    ATTRACT_GAIN : Hyper Parameter 
    REPULSIVE_GAIN : Hyper Parameter 
    Q_STAR : Hyper parameter
    """

    x_coor = np.arange(XLIMIN, XLIMAX, 0.25)
    y_coor = np.arange(XLIMIN, XLIMAX, 0.25)
    Y, X = np.meshgrid(x_coor, y_coor)
    nCoordsX = x_coor.shape[0]
    nCoordsY = y_coor.shape[0]

    U = np.zeros((nCoordsX, nCoordsY))
    points = []

    for i_x, x in enumerate(x_coor):
        for i_y, y in enumerate(y_coor):
            state = np.array([x, y])
            ptentl = potential(state, obs, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR, q_goal)
            U[i_y, i_x] = ptentl
            points.append((x, y))

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # x_len = x_coor.shape[0]
    # y_len = y_coor.shape[0]
    # ax.plot_surface(X, Y, U)
    # plt.show()

    return U, points

def gradientDescent(x, y, obs, q_start, q_goal, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR):
    """
    Gradient Descent using the Potential Field generated at the top
    x: Numpy array of the possible x direction of the point robot
    y: Numpy array of the possible y direction of the point robot 
    """
    
    # Tolerance 
    minimaTol = 1e-4

    # max iterations
    num_iterations = 3000000

    # Calculates the Potential Field 
    U, points = calc_potential_field(q_start, q_goal, obs, 0.25, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
    print("Entering Into Gradient Descent ")
    routes = []
    state = q_start
    currX, currY = 0,0
    
    # Appends the Route, state of the robot 
    routes.append([currX, currY])

    # calculate the gradient on the CSpace grid 
    dy, dx = np.gradient(U)

    # Part of gradient descent implementation
    dx_values = dx.flatten(order='F')
    dy_values = dy.flatten(order='F')

    # Gradeinte Desent Parameters 
    iterCount = 0
    updateRate = 20

    # Gradeint Descent Implementation includes the 
    while not isAtGoal(np.array([currX, currY]), q_goal):


        interp_gradient = np.zeros((2, 1))
        currX, currY = state

        # For Smooth Gradient Descent 
        interp_dx = griddata(points, dx_values, (currX, currY), method='cubic')
        interp_dy = griddata(points, dy_values, (currX, currY), method='cubic')
        interp_gradient[0] = 0.005 * interp_dx
        interp_gradient[1] = 0.005 * interp_dy

        print('state: ', state, 'dx: ', interp_dx, 'dy:', interp_dy)

        # Data Logging 
        shouldPrint = (iterCount % updateRate == 0)
        if shouldPrint:
            routes.append([round(currX[0], 1), round(currY[0], 1)])

        iterCount += 1
        
        # Updates the Status based on the gradeint 
        state -= interp_gradient
        currX, currY = state
        routes.append([currX[0], currY[0]])

        # Failure Conditions
        hitObstacle = any([np.isnan(stateCoord[0])
                            for stateCoord in state])
        atLocalMinima = ((np.linalg.norm(interp_gradient) < minimaTol) and
                            not isAtGoal(np.array([currX, currY])))
        outOfIterations = iterCount >= num_iterations

        if hitObstacle or atLocalMinima or outOfIterations:
            if(hitObstacle):
                print("Hit Obstacle")
            if(atLocalMinima):
                print("atLocalMinima")
            if(outOfIterations):
                print("outOfIterations")
            return False

    return routes

# Separate Functions had to be setup to implement used for the larger workspace
def calc_potential_field_large_workspace(q_start, q_goal, obs, NUMS, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR):
    """
    Calculates the potential of the all the states in the grid
    PARAMETERS
    ---------- 
    q_start : Start position of the point robot
    q_goal : Goal Position 
    obs : List of Shapely.Geometry.MultiPolygon 
    NUMS: Grid Size 
    DSTARGOAL : Hyper parameter
    ATTRACT_GAIN : Hyper Parameter 
    REPULSIVE_GAIN : Hyper Parameter 
    Q_STAR : Hyper parameter
    """
    x_coor = np.arange(XLIMIN, XLIMAX, 0.25)
    y_coor = np.arange(XLIMIN, XLIMAX, 0.25)
    Y, X = np.meshgrid(x_coor, y_coor)
    # potential = 0
    nCoordsX = len(x_coor)
    nCoordsY = len(y_coor)

    U = np.zeros((nCoordsX, nCoordsY))
    points = []

    for i_x, x in enumerate(x_coor):
        for i_y, y in enumerate(y_coor):
            state = np.array([x, y])
            ptentl = potential(state, obs, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR, q_goal)
            U[i_y, i_x] = ptentl
            points.append((x, y))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_len = x_coor.shape[0]
    y_len = y_coor.shape[0]
    ax.plot_surface(X, Y, U)
    plt.show()

    return U, points

def gradientDescent_field_large_workspace(x, y, obs, q_start, q_goal, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR):
    """
    Gradient Descent using the Potential Field generated at the top
    x: Numpy array of the possible x direction of the point robot
    y: Numpy array of the possible y direction of the point robot 
    """
    # Tolerance 
    minimaTol = 1e-4

    # max iterations
    num_iterations = 3000000

    # robot = self.robot
    U, points = calc_potential_field_large_workspace(q_start, q_goal, obs, 0.25,\
         DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
    print("Entering Into Gradient Descent ")
    routes = []
    state = q_start
    currX, currY = 0,0
    
    # Appends the Route, state of the robot 
    routes.append([currX, currY])

    # calculate the gradient on the CSpace grid
    dy, dx = np.gradient(U)

    dx_values = dx.flatten(order='F')
    dy_values = dy.flatten(order='F')

    # Initiation
    iterCount = 0
    updateRate = 20
    while not isAtGoal(np.array([currX, currY]), q_goal):

        interp_gradient = np.zeros((2, 1))
        currX, currY = state

        # For Smooth Gradient Descent 
        interp_dx = griddata(points, dx_values, (currX, currY), method='cubic')
        interp_dy = griddata(points, dy_values, (currX, currY), method='cubic')
        interp_gradient[0] = 0.002 * interp_dx
        interp_gradient[1] = 0.002 * interp_dy

        print('state: ', state, 'dx: ', interp_dx, 'dy:', interp_dy)

        # Data Logging
        shouldPrint = (iterCount % updateRate == 0)
        if shouldPrint:
            routes.append([round(currX[0], 1), round(currY[0], 1)])

        # Updates State
        iterCount += 1

        state -= interp_gradient
        currX, currY = state
        routes.append([currX[0], currY[0]])

        # Failure Conditon
        hitObstacle = any([np.isnan(stateCoord[0])
                            for stateCoord in state])
        atLocalMinima = ((np.linalg.norm(interp_gradient) < minimaTol) and
                            not isAtGoal(np.array([currX, currY])))
        outOfIterations = iterCount >= num_iterations

        if hitObstacle or atLocalMinima or outOfIterations:
            if(hitObstacle):
                print("Hit Obstacle")
            if(atLocalMinima):
                print("atLocalMinima")
            if(outOfIterations):
                print("outOfIterations")
            return False

    return routes

def plotPotentialField():

    c = input("Enter '0' for 2-obstacle(config 1) , '1' for 5-obstacle(config 2) , '2' for 9-obstacle(config 3): ")
    c = int(c)
    
    if c==0:
        # routes = [[0, 0], [1.98959, -0.00197], [3.22019, -0.57861], [4.74161, -0.84644], [5.6343, -0.7471], \
        # [5.74897,-0.60972 ], [5.70093, -0.33053], [5.81752,-0.13003], [6.07758, 0.17838], [6.65898, 0.32788], \
        #     [7.35296, 0.29694], [8.19067, 0.38579], [8.5821, 0.29594], [8.89755, 0.23071], [9.14425,0.17859 ],\
        #         [9.48265, 0.10806], [9.59782, 0.08396], [9.68742, 0.0652]]
        DSTARGOAL  = 8 
        ATTRACT_GAIN = 10 
        REPULSIVE_GAIN = [100,100]
        Q_STAR = [2, 1]
        XLIMIN = -10
        XLIMAX = 40
        NUMS = 50
        obs = WORKSPACE_CONFIG['WO3']
        q_start = WORKSPACE_CONFIG['start_pos']
        q_goal = WORKSPACE_CONFIG['WO3_goal']
        q_start = q_start.tolist()
        q_goal = q_goal.tolist()

        U, points = calc_potential_field(q_start, q_goal, obs, NUMS, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
        x = np.arange(XLIMIN, XLIMAX, 0.25)
        y = np.arange(XLIMIN, XLIMAX, 0.25)
        routes = gradientDescent(x, y, obs, q_start, q_goal, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
        print("ROUTES")
        print(routes)
        potField = U

        fig = plt.figure()
        ax = fig.add_subplot(111)

        xGrid = yGrid = NUMS

        x = np.arange(XLIMIN, XLIMAX, 0.25)
        y = np.arange(XLIMIN, XLIMAX, 0.25)
        dy, dx = np.gradient(potField)

        N = 50

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axisbelow(True)

        cs = ax.contourf(x, y, potField, N, alpha=0.7)
        plt.quiver(x, y, dx, dy, units='width')
        cbar = fig.colorbar(cs, ax=ax, orientation="vertical")
        cbar.ax.set_ylabel('potential function value')

        for obst in obs:
            x,y = obst.exterior.xy
            plt.plot(x,y, color='black', linewidth=0.5)


        val_x = [x[0] for x in routes]
        val_y = [y[1] for y in routes]
        plt.plot(val_x, val_y, color='red', marker='*', linestyle='none', linewidth=1, markersize=0.2, label='Robot path')

        # plotting the start / end location of the robot
        plt.plot(q_start[0], q_start[1],
                    color='green', marker='o', linestyle='none',
                    linewidth=2, markersize=1,
                    label='Starting State')

        plt.plot(q_goal[0], q_goal[1],
                    color='red', marker='x', linestyle='none',
                    linewidth=4, markersize=4,
                    label='Goal State')


        ax.set_aspect('equal')
        ax.set_xlim(-20, 40)
        ax.set_ylim(-20, 40)

        fig = plt.gcf()
        fig.show()
        plt.show()

    elif c==1:
        # routes = [[0, 0], [0.47, 0.47], [0.65, 0.65], [0.71, 0.64], [0.82, 0.62], [1.35, 0.56] ,[0.89, 0.61],[1.09, 0.59], [1.43, 0.88], [1.93, 1.2] ,[2.27, 1.63],[3.26, 2.5], [4.04, 3.35], [4.88, 4.11],[4.98, 5.39], [5.42, 5.81], [6.17, 7.04], [7.56, 8.12],[8.35, 8.73] ,[8.88, 9.14], [9.38, 9.52], [9.58, 9.67], [9.71, 9.78], [9.76, 9.82]]
        DSTARGOAL = 3 
        ATTRACT_GAIN = 9.2 
        REPULSIVE_GAIN = [2, 2, 2, 2, 2]
        Q_STAR = [2.7, 0.5, 1.2, 1.2, 5]
        XLIMIN = -10
        XLIMAX = 40
        NUMS = 50
        obs = WORKSPACE_CONFIG['WO1']
        q_start = WORKSPACE_CONFIG['start_pos']
        q_goal = WORKSPACE_CONFIG['WO1_goal']
        q_start = q_start.tolist()
        q_goal = q_goal.tolist()

        U, points = calc_potential_field(q_start, q_goal, obs, NUMS, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
        x = np.arange(XLIMIN, XLIMAX, 0.25)
        y = np.arange(XLIMIN, XLIMAX, 0.25)
        routes = gradientDescent(x, y, obs, q_start, q_goal, DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
        print("ROUTES")
        print(routes)
        potField = U

        fig = plt.figure()
        ax = fig.add_subplot(111)

        xGrid = yGrid = NUMS

        x = np.arange(XLIMIN, XLIMAX, 0.25)
        y = np.arange(XLIMIN, XLIMAX, 0.25)
        dy, dx = np.gradient(potField)

        N = 50

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axisbelow(True)

        cs = ax.contourf(x, y, potField, N, alpha=0.7)
        plt.quiver(x, y, dx, dy, units='width')
        cbar = fig.colorbar(cs, ax=ax, orientation="vertical")
        cbar.ax.set_ylabel('potential function value')

        for obst in obs:
            x,y = obst.exterior.xy
            plt.plot(x,y, color='black', linewidth=0.5)


        val_x = [x[0] for x in routes]
        val_y = [y[1] for y in routes]
        plt.plot(val_x, val_y, color='red', marker='*', linestyle='none', linewidth=1, markersize=0.2, label='Robot path')

        # plotting the start / end location of the robot
        plt.plot(q_start[0], q_start[1],
                    color='green', marker='o', linestyle='none',
                    linewidth=2, markersize=1,
                    label='Starting State')

        plt.plot(q_goal[0], q_goal[1],
                    color='red', marker='x', linestyle='none',
                    linewidth=4, markersize=4,
                    label='Goal State')


        ax.set_aspect('equal')
        ax.set_xlim(-20, 40)
        ax.set_ylim(-20, 40)

        fig = plt.gcf()
        fig.show()
        plt.show()

    elif c==2:
        DSTARGOAL  = 10 #6
        ATTRACT_GAIN = 0.1 #5
        REPULSIVE_GAIN = [2, 2, 7, 100 ,7, 7, 7, 7, 5, 5]
        Q_STAR = [4, 4, 6, 10 ,0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        XLIMIN = -10
        XLIMAX = 40
        NUMS = 40
        obs = WORKSPACE_CONFIG['WO2']
        q_start = WORKSPACE_CONFIG['start_pos']
        q_goal = WORKSPACE_CONFIG['WO2_goal']
        q_start = q_start.tolist()
        q_goal = q_goal.tolist()

        U, points = calc_potential_field_large_workspace(q_start, q_goal, obs, NUMS, DSTARGOAL,\
             ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
        x = np.arange(XLIMIN, XLIMAX, 0.25)
        y = np.arange(XLIMIN, XLIMAX, 0.25)
        routes = gradientDescent_field_large_workspace(x, y, obs, q_start, q_goal,\
             DSTARGOAL, ATTRACT_GAIN, REPULSIVE_GAIN, Q_STAR)
        print("ROUTES")
        print(routes)
        potField = U

        fig = plt.figure()
        ax = fig.add_subplot(111)

        xGrid = yGrid = NUMS

        x = np.arange(XLIMIN, XLIMAX, 0.25)
        y = np.arange(XLIMIN, XLIMAX, 0.25)
        dy, dx = np.gradient(potField)

        N = 50

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axisbelow(True)

        cs = ax.contourf(x, y, potField, N, alpha=0.7)
        plt.quiver(x, y, dx, dy, units='width')
        cbar = fig.colorbar(cs, ax=ax, orientation="vertical")
        cbar.ax.set_ylabel('potential function value')

        for obst in obs:
            x,y = obst.exterior.xy
            plt.plot(x,y, color='black', linewidth=0.5)


        val_x = [x[0] for x in routes]
        val_y = [y[1] for y in routes]
        plt.plot(val_x, val_y, color='red', marker='*', linestyle='none', linewidth=1, markersize=0.2, label='Robot path')

        # plotting the start / end location of the robot
        plt.plot(q_start[0], q_start[1],
                    color='green', marker='o', linestyle='none',
                    linewidth=2, markersize=1,
                    label='Starting State')

        plt.plot(q_goal[0], q_goal[1],
                    color='red', marker='x', linestyle='none',
                    linewidth=4, markersize=4,
                    label='Goal State')


        ax.set_aspect('equal')
        ax.set_xlim(-20, 40)
        ax.set_ylim(-20, 40)

        fig = plt.gcf()
        fig.show()
        plt.show()


def main():
    """
    Plots the Path and Vector of the Potential Gradient as a Color Plot 
    # Fair Warning 
    # Takes a ridiculous amount of time for gradient descent, additonally the gradient descent update had to be kept 
    # Really low (alpha) for optimum results but at the expense of time (a lot of time, in hours)
    """
    start = timer()
    plotPotentialField()
    finish = timer()
    computationTime = finish - start

    print("Computation TIme")
    print(computationTime)


if __name__ == '__main__':
    main()