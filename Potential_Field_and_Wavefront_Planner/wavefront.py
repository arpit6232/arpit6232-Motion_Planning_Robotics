"""
Implementation of the Wave Front , brush fire Algorithm

I would like to thank Professor Morteza Lahijanian and the fellow course mate Kedar More for 
discussion during the Implementation of this code 

"""
from timeit import default_timer as timer
import numpy as np
from math import pi
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from workspaces import config


WORKSPACE_CONFIG = config()

# Simulation parameters
limit = 200

class Robot(object):
    """
    This Class is helper class for plotting and mainipulating which keeps track the end points. 
    @param - arm_lengths : Array of the Lengths of each arm 
    @param - motor_angles : Current Angle of the Each Revolute Joint in the Global Frame
    """
    def __init__(self, arm_lengths, motor_angles):
        # Initialization with a specific parameter
        self.arm_lengths = np.array(arm_lengths)
        self.motor_angles = np.array(motor_angles)
        self.link_end_pts = [[0, 0], [0, 0], [0, 0]]
        # Find the Location of End Points of each Link
        for i in range(1, 3):
            # Follows Forward Kinematic Update Steps Analysis 
            self.link_end_pts[i][0] = self.link_end_pts[i-1][0] + self.arm_lengths[i-1] * \
                np.cos(np.sum(self.motor_angles[:i]))
            self.link_end_pts[i][1] = self.link_end_pts[i-1][1] + self.arm_lengths[i-1] * \
                np.sin(np.sum(self.motor_angles[:i]))

        self.end_effector = np.array(self.link_end_pts[2]).T

    def update_joints(self, motor_angles):
        """
        Update the Location of the end points of the link, Based on Updates of the End points 
        """
        self.motor_angles = motor_angles
        # Forward Kinematic Update and storage of link_length data
        for i in range(1, 3):
            self.link_end_pts[i][0] = self.link_end_pts[i-1][0] + self.arm_lengths[i-1] * \
                np.cos(np.sum(self.motor_angles[:i]))
            self.link_end_pts[i][1] = self.link_end_pts[i-1][1] + self.arm_lengths[i-1] * \
                np.sin(np.sum(self.motor_angles[:i]))

        self.end_effector = np.array(self.link_end_pts[2]).T

def make_grid(C_xSpace, C_ySpace, q_goal, obs):
    """
    Initialization of the Grid of specified Grid Size 
    PARAMETERS
    ----------
    C_xSpace : Numpy Array of the Possible X locations 
    C_ySpace : Numpy Array of the Possible Y locations 
    q_goal : Goal Location 
    obs : List of Shapely.Geometry.MultiPolygon 
    """
    gridX = len(C_xSpace) 
    gridY = len(C_ySpace) 

    # Initializing the Grid to 0
    grid=np.zeros((gridX,gridY))

    for i in range(gridX):
        for j in range(gridY):
            if (C_xSpace[i] == q_goal[0]) and (C_ySpace[j] == q_goal[1]):
                grid[i][j]=2
            
            for obstacle in obs:
                distToObst = abs(obstacle.exterior.distance(Point((C_xSpace[i], C_ySpace[j]))))
                if distToObst == 0:
                    grid[i][j]=1

    return grid

def boundary_condition_helper(grid, curr_status_num, i, j, gridX, gridY):
    """
    Helper Function to update grid values based on the curr_states of the grid value 
    PARAMETERS
    ----------
    grid : np.meshgrid 
    curr_status_num : Current Grid Value 
    i : X location 
    j : Y location 
    gridX : Length of the GRid X direction 
    gridY : Length of the GRid Y direction 
    """

    # Helper Variables
    rows = [i+1, i-1]
    cols = [j+1, j-1]

    # Keeps it within bounds 
    temp1 = rows[0] % gridX
    temp2 = cols[0] % gridY

    # For sanity check
    temp3 = max(rows[1],0)
    temp4 = max(cols[1],0)
    temp5 = curr_status_num + 1
    
    # Horizontal Facets
    if grid[temp1][j]==0 and rows[0] < gridX:
        grid[rows[0]][j] = temp5
    if grid[temp3][j]==0 and rows[1]:
        grid[rows[1]][j] = temp5

    # Vertical Facets
    if grid[i][temp2] == 0 and cols[0] < gridY:
        grid[i][cols[0]] = temp5
    if grid[i][temp4] == 0 and cols[1] :
        grid[i][cols[1]] = temp5

    return grid

def grid_update(grid,curr_grid_number, C_xSpace, C_ySpace, q_start):
    """
    Updates the grid Based on the current status of the grid
    PARAMETERS
    ----------
    grid : np.meshgrid 
    curr_grid_number: Current Value of the grid value 
    C_xSpace : Numpy Array of the Possible X locations 
    C_ySpace : Numpy Array of the Possible Y locations 
    q_start : Start Location 
    """

    # Initialization 
    gridX = len(C_xSpace)
    gridY = len(C_ySpace)


    # O(n2) implementation to dig through all the grid values 
    for i in range(gridX):
        for j in range(gridY):
            if grid[i][j]==curr_grid_number:
                # Success Condition 
                if C_xSpace[i] == q_start[0][0] and C_ySpace[j] == q_start[1][0]:
                    return grid, curr_grid_number
                else:
                    # Continue Updating The Grid 
                    boundary_condition_helper(grid, curr_grid_number, i, j, gridX, gridY)
    return grid, None

def planning_path(grid,curr_grid_number, C_xSpace, C_ySpace):
    """
    Finds the Route Post the Grid Update 
    PARAMETERS
    ----------
    grid : np.meshgrid 
    curr_grid_number : Current Grid Number 
    C_xSpace : Numpy Array of the Possible X locations 
    C_ySpace : Numpy Array of the Possible Y locations 
    """
    # Initial Flag Setup 
    flag=curr_grid_number+10

    # Finds the Start Location 
    i=np.argwhere(C_xSpace==0)[0][0]
    j=np.argwhere(C_ySpace==0)[0][0]

    # Updates the Start for approprate flag 
    # Also necessary to showcase output as 
    # plt.imshow() is used 
    grid[(i)][j]=flag
    
    # Begin Calculation 
    distance=0
    
    # While not Goal is reached 
    while curr_grid_number>2:
        status = False
        # Right
        if grid[(i+1)][j]==curr_grid_number-1:
            i+=1
            status = True
        
        # Left 
        elif grid[(i-1)][j]==curr_grid_number-1:
            i-=1
            status = True

        # Top
        elif grid[(i)][j+1]==curr_grid_number-1:
            j+=1
            status = True

        # Bottom 
        elif grid[(i)][j-1]==curr_grid_number-1:
            j-=1
            status = True

        # Update Appropriate Flags 
        if status:
            # Grid size is 0.5
            distance+=0.25
            grid[(i)][j]=flag
            # Update the grid number 
            curr_grid_number-=1

    return distance


# Manipulator
def manipulator_cspace(C_xSpace, C_ySpace, q_goal, obs):
    """
    Gives the Grid of C Space Generated for a 2 link manipulator 
    PARAMETERS
    ----------
    C_xSpace : Numpy Array of the Possible X locations 
    C_ySpace : Numpy Array of the Possible Y locations 
    q_goal : Goal Location 
    obs : List of Shapely.Geometry.MultiPolygon 
    """
    # Initialization 
    gridX = len(C_xSpace)
    gridY = len(C_ySpace)
    
    # Initialize with 0's
    grid=np.zeros((int(gridX),int(gridY)))

    # Setup for link length as 1
    arm_lengths = [float(1.0), float(1.0)]
    # Begin Setup with motor angles to [0,0]
    motor_angles = np.array([0] * 2)

    obstacles = obs
    temp = []
    temp_status = False
    plt.ion()
    plt.show(block=False)
    arm = Robot(arm_lengths, motor_angles)

    # Rotate through all the possible angles of the Link
    theta_np_array = np.radians(np.arange(0, 365, 5))
    for i in range(gridX):
        for j in range(gridY):
            # Updates the Motor Joints for every 5 degree update
            arm.update_joints([theta_np_array[i], theta_np_array[j]])
            link_end_pts = arm.link_end_pts
            
            collision_detected = False

            # Checks if it intersects the obstacle 
            for k in range(len(link_end_pts) - 1):
                for obstacle in obstacles:
                    # Create a line segment 
                    line_seg = [link_end_pts[k], link_end_pts[k + 1]]
                    line = LineString([link_end_pts[k], link_end_pts[k + 1]])
                    collision_detected = line.intersects(obstacle)
                    if collision_detected:
                        break
                if collision_detected:
                    break
            # Updates it 1 if it intersects 
            grid[i][j] = int(collision_detected)
    
    # Hard Coding the goal location
    grid[36][0] = 2

    return grid

def boundary_condition_helper_manipulator(grid, curr_grid_number, i, j, gridX, gridY):
    """
    Helper Function to update grid values based on the curr_states of the grid value 
    PARAMETERS
    ----------
    grid : np.meshgrid 
    curr_status_num : Current Grid Value 
    i : X location 
    j : Y location 
    gridX : Length of the GRid X direction 
    gridY : Length of the GRid Y direction 
    """

    # Helper Variables
    rows = [i+1, i-1]
    cols = [j+1, j-1]

    # Keeps it within bounds of 0 and 360 degrees 
    # Resets 360 to 0 degrees 
    temp1 = (rows[0]) % (gridX-1)
    temp2 = (rows[1]) % (gridX-1)

    # For sanity check
    temp3 = (cols[0]) % (gridY-1)
    temp4 = (cols[1]) % (gridY-1)
    temp5 = curr_grid_number+1
    
    # Horizontal Facets
    if grid[temp1][j] == 0:
        grid[temp1][j] = temp5
    if grid[temp2][j] == 0:
        grid[temp2][j] = temp5

    # Vertical Facets
    if grid[i][temp3] == 0:
        grid[i][temp3] = temp5
    if grid[i][temp4] == 0:
        grid[i][temp4] = temp5

    return grid

def grid_update_manipulator(grid, curr_grid_number, C_xSpace, C_ySpace, q_start):
    """
    Updates the grid Based on the current status of the grid
    PARAMETERS
    ----------
    grid : np.meshgrid 
    curr_grid_number: Current Value of the grid value 
    C_xSpace : Numpy Array of the Possible X locations 
    C_ySpace : Numpy Array of the Possible Y locations 
    q_start : Start Location 
    """

    # Initialization 
    gridX = len(C_xSpace)
    gridY = len(C_ySpace)

    # O(n2) implementation to dig through all the grid values 
    for i in range(gridX):
        for j in range(gridY):
            if grid[i][j]==curr_grid_number:
                # Goal Conditon 
                if [round(C_xSpace[i],2), round(C_ySpace[j],2)] == [0.00,0.00] or \
                    [round(C_xSpace[i],2), round(C_ySpace[j],2)] == [6.28,6.28]:
                    return grid, curr_grid_number
                else:
                    # Boundary Conditon 
                    grid = boundary_condition_helper_manipulator(grid, curr_grid_number, i, j, gridX, gridY)

    plt.imshow(grid, origin = 'lower')
    plt.show()
    return grid, None

def planning_path_manipulator(grid,curr_grid_number, C_xSpace, C_ySpace):
    """
    Finds the Route Post the Grid Update 
    PARAMETERS
    ----------
    grid : np.meshgrid 
    curr_grid_number : Current Grid Number 
    C_xSpace : Numpy Array of the Possible X locations 
    C_ySpace : Numpy Array of the Possible Y locations 
    """

    # Val stores the Location 
    val = []
    gridX = len(C_xSpace)-1
    gridY = len(C_ySpace)-1

    # Initial Flag Setup 
    flag = curr_grid_number + 10

    # Start Location 
    i = 0
    j = 0

    # Updates the Start for approprate flag 
    # Also necessary to showcase output as 
    # plt.imshow() is used 
    grid[(i)][j]=flag

    # Ready, Set, Go 
    distance=0

    # While Goal is not reached 
    while curr_grid_number>2:
        status = False

        # Right 
        if grid[(i+1) % gridX][j]==curr_grid_number-1:
            status = True
            i = (i+1) % gridX

        # Left
        elif grid[(i-1) % gridX][j]==curr_grid_number-1:
            status = True
            i = (i-1) % gridX

        # Top    
        elif grid[(i)][(j+1) % gridY]==curr_grid_number-1:
            status = True
            j = (j+1) % gridY
        
        # Bottom 
        elif grid[(i)][(j-1) % gridY]==curr_grid_number-1:
            status = True
            j = (j-1) % gridY

        # Update Flags 
        if(status):
            # In degrees 
            distance += 5
            val.append([i, j])
            grid[(i)][j]=flag
            plt.imshow(grid, origin = 'lower')
            plt.show()
            # Update the grid number 
            curr_grid_number -= 1

    return distance, val

def forw_K(motor_angles):
    """
    Function to Calculate the forward kinematics.
    """
    pos_x_link1 = 0
    pos_y_link1 = 0
    pos_x = 0
    pos_y = 0
    # Simple logic gets the calculates the End Effector position 
    # from the current motor angle and position
    for i in range(1, 3):
        pos_x += 1.0 * np.cos(np.sum(motor_angles[:i]))
        pos_y += 1.0 * np.sin(np.sum(motor_angles[:i]))

    pos_x_link1 += 1.0 * np.cos(np.sum(motor_angles[:1]))
    pos_y_link1 += 1.0 * np.sin(np.sum(motor_angles[:1]))


    # Transpose is necessary, for future updates
    return pos_x_link1, pos_y_link1, pos_x, pos_y

def manipulator_plotter(val, obs, q_goal):
    """
    Helper Function to Plot the 2 link manipulator 
    """
    plt.figure(2)
    for idx, angle in enumerate(val):
        # Plots every 10 iteration
        if(idx % 10 == 0):
            angle = np.asarray(angle)*5
            mtr_angles = np.radians(angle)

            # Results from Forward Kinematics
            pos_x_link1, pos_y_link1, pos_x_link2, pos_y_link2 = forw_K(mtr_angles)

            plt.plot([0.0, pos_x_link1, pos_x_link2],\
                            [0.0, pos_y_link1, pos_y_link2], 'c-')

            plt.plot(pos_x_link1, pos_y_link1, 'ko', markersize=2)
            plt.plot(pos_x_link2, pos_y_link2, 'ro')

            for obstacle in obs:
                plt.plot(*obstacle.exterior.xy)

    # Mark the goal Position
    plt.plot(q_goal[0],q_goal[1], 'rx')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.draw()
    plt.pause(100)
    plt.show()

def main():
    start = timer()

    m1 = input("0 for non-manipulator , 1 or Manipulator : ")

    if(int(m1) == 0):
    # # Non- Manipulator
        cfg = input(" Enter 1 for config 1 , 2 for config 2: ")
        cfg = int(cfg)
        if cfg == 1:
            XLIMIN = -10
            XLIMAX = 40
            YLIMIN = -20
            YLIMAX = 20
            obs = WORKSPACE_CONFIG['WO1']
            q_start = WORKSPACE_CONFIG['start_pos']
            q_goal = WORKSPACE_CONFIG['WO1_goal']
            q_start = q_start.tolist()
            q_goal = q_goal.tolist()

            C_xSpace = np.arange(XLIMIN, XLIMAX, 0.25)
            C_ySpace = np.arange(XLIMIN, XLIMAX, 0.25)
            grid = make_grid(C_xSpace, C_ySpace, q_goal, obs)
            fig,ax=plt.subplots()
            curr_grid_number=2
            while True:
                updated_grid, val =grid_update(grid,curr_grid_number, C_xSpace, C_ySpace, q_start)
                if val is not None:
                    # Success Condition
                    grid,curr_grid_number=updated_grid, val
                    break
                else:
                    # Continue to Update GRid 
                    grid=updated_grid
                    curr_grid_number=curr_grid_number+1
                    pass


            dist=planning_path(grid,curr_grid_number, C_xSpace, C_ySpace)
            print("Total distace of the path is:",dist)
            plt.imshow(grid, origin='lower')
            plt.show()

        elif cfg == 2:
            XLIMIN = -10
            XLIMAX = 40
            YLIMIN = -8
            YLIMAX = 8
            obs = WORKSPACE_CONFIG['WO2']
            q_start = WORKSPACE_CONFIG['start_pos']
            q_goal = WORKSPACE_CONFIG['WO2_goal']
            q_start = q_start.tolist()
            q_goal = q_goal.tolist()

            C_xSpace = np.arange(XLIMIN, XLIMAX, 0.25)
            C_ySpace = np.arange(XLIMIN, XLIMAX, 0.25)
            grid = make_grid(C_xSpace, C_ySpace, q_goal, obs)
            fig,ax=plt.subplots()
            curr_grid_number=2
            while True:
                updated_grid, val =grid_update(grid,curr_grid_number, C_xSpace, C_ySpace, q_start)
                if val is not None:
                    # Success Condition
                    grid,curr_grid_number=updated_grid, val
                    break
                else:
                    # Continue to Update GRid 
                    grid=updated_grid
                    curr_grid_number=curr_grid_number+1
                    pass


            dist=planning_path(grid,curr_grid_number, C_xSpace, C_ySpace)
            print("Total distace of the path is:",dist)
            plt.imshow(grid.T, origin='lower')
            plt.show()

    else: 
        # Manipulator 
        obs = WORKSPACE_CONFIG['WO4']
        q_start = WORKSPACE_CONFIG['manip_start_pos']
        q_goal = WORKSPACE_CONFIG['manip_goal_pos']
        q_start = q_start.tolist()
        q_goal = q_goal.tolist()

        C_xSpace = np.arange(0, 365, 5)
        C_ySpace = np.arange(0, 365, 5)
        grid = manipulator_cspace(C_xSpace, C_ySpace, q_goal, obs)
        
        fig,ax = plt.subplots()
        curr_grid_number = 2
        while True: 
            updated_grid, val = grid_update_manipulator(grid, curr_grid_number, C_xSpace, C_ySpace, q_start)
            if val is not None:
                    # Success Condition
                    grid,curr_grid_number=updated_grid, val
                    break
            else:
                # Continue to Update GRid 
                grid=updated_grid
                curr_grid_number=curr_grid_number+1
                pass

        dist, val = planning_path_manipulator(grid,curr_grid_number, C_xSpace, C_ySpace)
        # plt.pause(100)
        manipulator_plotter(val, obs, q_goal)
        print("Total distace of the path is:",dist)
        plt.imshow(grid, origin = 'lower')
        plt.show()

    finish = timer()
    computationTime = finish - start

    print("Computation TIme")
    print(computationTime)


if __name__ == '__main__':
    main()