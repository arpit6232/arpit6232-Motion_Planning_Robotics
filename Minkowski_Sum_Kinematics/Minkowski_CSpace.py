import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from math import sin, cos, radians
import scipy
import pylab
from mpl_toolkits.mplot3d import Axes3D



def rotate_point(point, angle, center_point=(0, 0)):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * cos(angle_rad) - new_point[1] * sin(angle_rad),
                 new_point[0] * sin(angle_rad) + new_point[1] * cos(angle_rad))
    # Reverse the shifting we have done
    new_point = (new_point[0] + center_point[0], new_point[1] + center_point[1])
    return new_point


def rotate_polygon(polygon, angle, center_point=(0, 0)):
    """Rotates the given polygon which consists of corners represented as (x,y)
    around center_point (origin by default)
    Rotation is counter-clockwise
    Angle is in degrees
    """
    rotated_polygon = []
    for corner in polygon:
        rotated_corner = rotate_point(corner, angle, center_point)
        rotated_polygon.append(rotated_corner)
    return rotated_polygon


def rotate_at_angle(robot, angle):
    centroid_x = sum([x[0] for x in robot])/3
    centroid_y = sum([y[1] for y in robot])/3
    return rotate_polygon(robot, angle, (centroid_x,centroid_y))


def minkowski_sum(obstacle, robot):
    ms = []
    res = []
    for i in range(len(obstacle)):
        for j in range(len(robot)):
            ms.append(( obstacle[i][0] + robot[j][0] , obstacle[i][1] + robot[j][1]))
    ms.sort()
    for pts in ms:
        if pts not in res:
            res.append(pts)
    return res


def main():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    obstacle = [(0, 0), (0, 2), (1, 2)]
    levels = input("Enter Number of Levels to see in the 3d Plots between 0 and 360\n")
    robot = [(-1.0*x[0], -1.0*x[1]) for x in obstacle]
    r_val = np.linspace(0, 360, int(levels))
    final_CSpace = []
    for r in r_val:
        CSpace = minkowski_sum(obstacle, robot)
        points = np.array([*CSpace])
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            ax.plot3D(points[simplex, 0], points[simplex, 1], r, 'b-')
        robot = rotate_at_angle(robot, r)

    plt.show()



if __name__ == '__main__':
    main()