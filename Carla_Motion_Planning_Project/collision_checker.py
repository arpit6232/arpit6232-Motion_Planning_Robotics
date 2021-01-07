#!/usr/bin/env python3

import numpy as np
import scipy.spatial
from math import sin, cos, pi, sqrt

"""
Linked Research Paper: . Tan, C. Liu and Y. Yu, "The Design of Collision Detection Algorithm in 2D Grapple 
                        Games," 2009 International Conference on Information Engineering and Computer
                        Science, Wuhan, 2009, pp. 1-4, doi: 10.1109/ICIECS.2009.5363537.

2) Concepts Acquired from:  https://www.coursera.org/specializations/self-driving-cars
"""

class CollisionChecker:
    def __init__(self, c_offs, c_r, w):
        self._c_offs = c_offs
        self._c_r   = c_r
        self._w         = w

    def collision_check(self, paths, obstacles):
        """Returns a bool array on whether each path is collision free.
        PARAMETERS
        ----------
            paths: A list of paths in the global frame.  
            obstacles: A list of [x, y] points that represent points along the
                border of obstacles, in the global frame
        RETURNS
        -------
            coll_arr: A list of boolean values which classifies
                whether the path is collision-free (true)
        """
        coll_arr = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            flag = True
            path           = paths[i]

            # Points in Path
            for j in range(len(path[0])):
                c_loc = np.zeros((len(self._c_offs), 2))
                c_off = np.array(self._c_offs)
                c_loc[:, 0] = path[0][j] + c_off * cos(path[2][j])
                c_loc[:, 1] = path[1][j] + c_off * sin(path[2][j])
                for k in range(len(obstacles)):
                    coll_dists = \
                        scipy.spatial.distance.cdist(obstacles[k], 
                                                     c_loc)
                    coll_dists = np.subtract(coll_dists, self._c_r)
                    flag = flag and not np.any(coll_dists < 0)

                    if not flag:
                        break
                if not flag:
                    break

            coll_arr[i] = flag

        return coll_arr

    def select_best_path_index(self, paths, coll_arr, g_s):
        """Returns the path index which is best suited for the vehicle to
        traverse.
        Selects a path index which is closest to the center line as well as far
        away from collision paths.
        PARAMETERS
        ----------
            paths: A list of paths in the global frame.
            coll_arr: A list of boolean values which classifies
                whether the path is collision-free (true).
            g_s: Goal state for the vehicle to reach (centerline goal).
                format: [x_goal, y_goal, v_goal]
        RETURNS
        -------
            best_idx: The path index which is best suited for the vehicle to
                navigate with.
        """
        best_idx = None
        best_score = float('Inf')
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if coll_arr[i]:
                # Compute the "distance from centerline" score.
                score = np.sqrt((paths[i][0][-1]-g_s[0])**2+(paths[i][1][-1]-g_s[1])**2)

                for j in range(len(paths)):
                    if j == i:
                        continue
                    else:
                        if not coll_arr[j]:
                            pass

            # Colliding paths
            else:
                score = float('Inf')
                
            # Best Index
            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx