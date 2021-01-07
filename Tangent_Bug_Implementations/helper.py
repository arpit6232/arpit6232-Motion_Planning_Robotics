import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import time 
import polytope as pc 
import scipy
from scipy.spatial import Delaunay as Del
from workspaces import config
from operator import eq, add, sub

WORKSPACE_CONFIG = config()

def min_index(pts):
    """
    PARAMETERS
    ----------
    Vertexes of Polytopes : List

    RETURNS
    -------
    Index of the Vertex
    """
    min = np.NINF
    status= None 
    for idx, val in enumerate(pts):
        if val > min:
            min = val 
            status = idx 
    
    return status

def polytopeEuclideanDistance(pts, goal):
    """
    PARAMETERS
    ----------
    Vertexes of Polytopes : List

    RETURNS
    -------
    Smallest Distance from the the vertex of polytopes and the goal
    """
    min_val = np.Inf
    stat = None 
    for idx, pt in enumerate(pts):
        dist = np.sqrt( (pt[0] - goal[0][0])**2 + (pt[1] - goal[1][0])**2 )
        if dist < min_val:
            min_val = dist 
            stat = idx
        
    return stat

def NextLocationXY(EnabledPolytope, limit_x, limit_y):
    """
    Helper Function to Vertexes of Polytope
    """
    vertex1 = vertex2  = None
    vertex2 = None

    for idx, vertex in enumerate(EnabledPolytope.vertices):
        if limit_x == vertex[0] and limit_y == vertex[1]:
            
            vertex1 = vertex
            if idx != 0:
                vertex2 = EnabledPolytope.vertices[idx - 1]
            else:
                vertex2 = EnabledPolytope.vertices[-1]
    return vertex1, vertex2


def NextPoseHelperFunc(EnabledPolytope, facet_vertex):

    vx = None
    for idx, row in enumerate(EnabledPolytope.vertices):
        if facet_vertex[0] == row[0] and facet_vertex[1] == row[1]:
            vx = idx
            break
    return vx


def PoseFromPolytopeVertex(x, y, EnabledPolytope, vflag, hflg):
    """
    Helper Function to locate the next possible pose Vertex from the current 
    pose of the robot for Bug 2 Algorithm 
    PARAMETERS
    ----------
        (x, y) : Current Pose of the Robot 
        EnabledPolytope: Currently active polytope 
        vflag: Flag to Check if Vertical Traversal is preferred 
        hflag: Flag to check is Horizontal Traversal is preferred 

    RETURNS
    -------
        Next Pose of the Robot
    """
    x = round(x)
    for idx, vtx in enumerate(EnabledPolytope.vertices):
        if x == vtx[0] or y == vtx[1]:
            flag = False
            pose = idx
            if x == vtx[0]:
                flag = True

            if vflag:
                for idx, vtx in enumerate(EnabledPolytope.vertices):
                    if x == vtx[0]:
                        flag = True
                        pose = idx
                        break
            if hflg:
                for idx, vtx in enumerate(EnabledPolytope.vertices):
                    if y == vtx[1] and x == vtx[0]:
                        flag = False
                        pose = idx
                        break
            break
    
    possible_pose = pose + 1
    if flag:
        if pose != 0:
            if (EnabledPolytope.vertices[pose - 1][0] == x) or (EnabledPolytope.vertices[pose - 1][1] == y) :
                possible_pose = pose - 1
        else:
            if (EnabledPolytope.vertices[3][0] == x) or (EnabledPolytope.vertices[3][1] == y):
                possible_pose = 3

    if possible_pose == 4:
        possible_pose = 0

    v1 = EnabledPolytope.vertices[pose]
    v2 = EnabledPolytope.vertices[possible_pose]

    if not flag:
        if x <= round(v1[0], 1):
            v1, v2 = v2, v1
        if not hflg:
            v1, v2 = v2, v1
    else:
        if (y > round(v1[1], 1)):
            v1, v2 = v2, v1
        if not vflag:
            v1, v2 = v2, v1

    return v2, v1

