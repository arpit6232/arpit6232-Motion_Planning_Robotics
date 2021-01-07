#!/usr/bin/env python3

"""
1) Linked Research Paper: M. Pivtoraiko and A. Kelly, ”Differentially constrained motion re-
planning using state lattices with graduated fidelity,” 2008 IEEE/RSJ
International Conference on Intelligent Robots and Systems, Nice, 2008,
pp. 2611-2616, doi: 10.1109/IROS.2008.4651220.

"""

import numpy as np
import copy
import path_optimizer
import collision_checker
import velocity_planner
from math import sin, cos, pi, sqrt

class LocalPlanner:
    def __init__(self, num_paths, p_off, c_off, c_r, 
                 p_weight, time_del, a_max, slow_speed, 
                 stop_line_buffer):
        self._num_paths = num_paths
        self._p_off = p_off
        self._path_optimizer = path_optimizer.PathOptimizer()
        self._collision_checker = \
            collision_checker.CollisionChecker(c_off,
                                               c_r,
                                               p_weight)
        self._velocity_planner = \
            velocity_planner.VelocityPlanner(time_del, a_max, slow_speed, 
                                             stop_line_buffer)

    # Computes the goal state set from a given goal position by lateral sampling offsets
    def possible_goal_set(self, g_idx, g_state, wps, e_s):
        """Gets the goal states given a goal position.
        PARAMETERS
        ----------
            1) g_idx: Goal index for the vehicle to reach
                i.e. wps[g_idx] gives the goal waypoint
            2) g_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal], in units [m, m, m/s]
            3) wps: current wps to track (global frame). 
                format: [[x0, y0, v0]]
            4) e_s: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
        RETURNS
        -------
            goal_state_set: Set of goal states to be used by the local planner to plan multiple
                proposal paths. This goal state set is in the vehicle frame.
                format: [[xm, ym, tm, vm]]
        """
        if g_idx == len(wps) - 1:
            del_x = wps[g_idx][0] - wps[g_idx-1][0]
            del_y = wps[g_idx][1] - wps[g_idx-1][1]
        else:
            del_x = wps[g_idx+1][0] - wps[g_idx][0]
            del_y = wps[g_idx+1][1] - wps[g_idx][1]
        heading = np.arctan2(del_y, del_x)

        g_state_curr = copy.copy(g_state)

        # Translate so the ego state is at the origin in the new frame.
        g_state_curr[0] -= e_s[0]
        g_state_curr[1] -= e_s[1]

        x, y = g_state_curr[0], g_state_curr[1]

        theta = -e_s[2]
        g_x = cos(theta) * x - sin(theta) * y
        g_y = sin(theta) * x + cos(theta) * y

        g_t = heading + theta

        # Velocity is preserved post the transformation.
        # g_v = g_state[2]

        if g_t > pi:
            g_t -= 2*pi
        elif g_t < -pi:
            g_t += 2*pi

        goal_state_set = []
        for i in range(self._num_paths):
            # Potential path to be considered by the local planner.
            offs = (i - self._num_paths // 2) * self._p_off
            offs_x = cos(g_t + pi/2) * offs
            offs_y = sin(g_t + pi/2) * offs

            goal_state_set.append([g_x + offs_x, 
                                   g_y + offs_y, 
                                   g_t, 
                                   g_state[2]])
           
        return goal_state_set  
              
    def trajectory_plan(self, goal_state_set):
        """Plans the path set using the polynomial spiral optimization.
        Plans the path set using polynomial spiral optimization to each of the
        goal states.
        PARAMETERS
        ----------
            1) goal_state_set: Set of goal states to be used by the local planner to plan multiple
                proposal paths. This goal state set is in the vehicle frame.
                format: [[xm, ym, tm, vm]]
        RETURNS
        -------
            1) paths: A list of optimized spiral paths which satisfies the set of 
                goal states. A path is a list of points of the following format:
                    [x_points, y_points, t_points]
            2) path_validity: List of booleans classifying whether a path is valid
                (true) or not (false) for the local planner to traverse. Each ith
                path_validity corresponds to the ith path in the path list.
        """
        paths         = []
        path_validity = []
        for g_state in goal_state_set:
            path = self._path_optimizer.optimize_spiral(g_state[0], 
                                                        g_state[1], 
                                                        g_state[2])
            if np.linalg.norm([path[0][-1] - g_state[0], 
                               path[1][-1] - g_state[1], 
                               path[2][-1] - g_state[2]]) > 0.1:
                path_validity.append(False)
            else:
                paths.append(path)
                path_validity.append(True)

        return paths, path_validity

def frame_adj(paths, e_s):
    """ Converts the to the global coordinate frame.
    Converts the paths from the local (vehicle) coordinate frame to the
    global coordinate frame.
    PARAMETERS
    ----------
        1) paths: A list of optimized spiral paths which satisfies the set of 
                goal states. A path is a list of points of the following format:
                    [x_points, y_points, t_points]
        2) e_s: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
    RETURNS
    -------
        transformed_paths: A list of transformed paths in the global frame.  
            A path is a list of points of the following format:
                [x_points, y_points, t_points]
    """
    transformed_paths = []
    for path in paths:
        _x = []
        _y = []
        _t = []

        for i in range(len(path[0])):
            _x.append(e_s[0] + path[0][i]*cos(e_s[2]) - \
                                                path[1][i]*sin(e_s[2]))
            _y.append(e_s[1] + path[0][i]*sin(e_s[2]) + \
                                                path[1][i]*cos(e_s[2]))
            _t.append(path[2][i] + e_s[2])

        transformed_paths.append([_x, _y, _t])

    return transformed_paths