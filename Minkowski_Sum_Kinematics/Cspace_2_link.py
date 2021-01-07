from math import pi
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geom
import descartes

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


def cspace():
    """
    Code expects the user to input to Input the Number of obstacles, the number of vertex inside the obstacle 
    and the coordinates of each obstacle. 

    Sample Solution of the following is attached with the folder 
    a)  a workspace with a triangular obstacle with vertices (0.25,0.25), (0,0.75), and (−0.25,0.25).
    b)  a workspace with two large rectangular obstacles with vertices:
        O1:    (−0.25,1.1),(−0.25,2),(0.25,2),and  (0.25,1.1),
        O2:    (−2,−2),(−2,−1.8),(2,−1.8),and  (2,−2).
    c)  a workspace with two obstacles:
        O1:    (−0.25,1.1),(−0.25,2),(0.25,2),and  (0.25,1.1),
        O2:    (−2,−0.5),(−2,−0.3),(2,−0.3),and  (2,−0.5)

    Ctrl+C is needed to exit the program 
    """
    polygons_list = list()
    vertex_list = list()
    l1 = input("Enter Length of Link 1\n")
    l2 = input("Enter Length of Link 2\n")
    arm_lengths = [float(l1), float(l2)]
    motor_angles = np.array([0] * 2)


    num_obs = int(input("Enter the number of obstacles: "))
    assert num_obs > 0
    for obs in range(num_obs):
        print("For Obstacle :", obs+1)
        num_vertex = int(input("Enter the number of Vertexes of Polygon: "))
        for v in range(num_vertex):
            print("For Vertex :", v+1)
            x = float(input("Enter the X coordinate of Vertex: "))
            y = float(input("Enter the Y coordinate of Vertex: "))
            vertex_list.append((x,y))
        polygons_list.append(geom.Polygon(vertex_list))
        vertex_list = []

    obstacles = geom.MultiPolygon(polygons_list)

    print("Plotting Configuration-Space")

    plt.ion()
    plt.show(block=False)
    arm = Robot(arm_lengths, motor_angles)

    #Subdivide the The plot in a 100 by 100 grid
    grid = [[0 for _ in range(limit)] for _ in range(limit)]
    theta_list = [2 * i * pi / limit for i in range(-limit // 2, limit // 2 + 1)]
    for i in range(limit):
        for j in range(limit):
            # Rotates the 2 link robot in the 
            arm.update_joints([theta_list[i], theta_list[j]])
            link_end_pts = arm.link_end_pts
            collision_detected = False
            for k in range(len(link_end_pts) - 1):
                for obstacle in obstacles:
                    line_seg = [link_end_pts[k], link_end_pts[k + 1]]
                    line = geom.LineString([link_end_pts[k], link_end_pts[k + 1]])
                    collision_detected = line.intersects(obstacle)
                    if collision_detected:
                        break
                if collision_detected:
                    break
            grid[i][j] = int(collision_detected)
    plt.imshow(grid)
    plt.pause(100)
    plt.show()


def main():
    cspace()


if __name__ == '__main__':
    main()
