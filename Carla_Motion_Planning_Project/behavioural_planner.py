#!/usr/bin/env python3

"""
1) Linked Research Paper: J. Wei, J. M. Snider, T. Gu, J. M. Dolan and B. Litkouhi,
 "A behavioral planning framework for autonomous driving," 2014 IEEE Intelligent Vehicles 
Symposium Proceedings, Dearborn, MI, 2014, pp. 458-464, doi: 10.1109/IVS.2014.6856582.
"""

import numpy as np
import math

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
STOP_THRESHOLD = 0.02
STOP_COUNTS = 10

class BehaviouralPlanner:
    def __init__(self, la, spf, lead_vehicle_la):
        """
        Hyper-parameters to tune
        """
        self._la = la
        self._spf = spf
        self._f_l_a_la = lead_vehicle_la
        self._state = 0
        self._f_l_a = False
        self._goal_state = [0.0, 0.0, 0.0]
        self._g_idx = 0
        self._stop_count = 0

    def set_lookahead(self, la):
        self._la = la

    """
    State Transition Function
    """

    def trans_state(self, wp, e_s, c_l_v):
        """Handles state transitions and computes the goal state.  
        
        PARAMETERS
        ----------
            1) wp: current waypoints to track (global frame). 
                format: [[x0, y0, v0]]
            2) e_s: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
            3) c_l_v: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._g_idx: Index of the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._stop_count: Iteration count to in STAY_STOPPED state.
        """
        if self._state == 0:
            # Closest index to the ego vehicle.
            closest_len, c_idx = get_c_idx(wp, e_s)

            # Goal index that lies within the lookahead distance along the waypoints.
            arc_length = closest_len
            wp_idx = c_idx
            
            # Closest waypoint is already far enough for the planner
            if arc_length > self._la:
                return wp_idx

            # End of path
            if wp_idx == len(wp) - 1:
                return wp_idx

            # Find next waypoint.
            num_wps = len(wp)
            for i in range(wp_idx + 1, num_wps):
                arc_length += math.sqrt(
                    (wp[i][0] - wp[i-1][0])**2 + (wp[i][1] - wp[i-1][1])**2
                )
                if arc_length > self._la:
                    break

            g_idx = i

            # Index set between closest_index and goal_index for stop signs
            stop_sign_found = False
            for i in range(c_idx, g_idx):
                coll = False
                for stp_sign_f in self._spf:

                    v1     = np.subtract(np.array(wp[i+1][0:2]), np.array(wp[i][0:2]))
                    v2     = np.subtract(np.array(stp_sign_f[0:2]), np.array(wp[i+1][0:2]))
                    flag_1 = np.sign(np.cross(v1, v2))
                    v2     = np.subtract(np.array(stp_sign_f[2:4]), np.array(wp[i+1][0:2]))
                    flag_2 = np.sign(np.cross(v1, v2))

                    v1     = np.subtract(np.array(stp_sign_f[2:4]), np.array(stp_sign_f[0:2]))
                    v2     = np.subtract(np.array(wp[i][0:2]), np.array(stp_sign_f[2:4]))
                    flag_3 = np.sign(np.cross(v1, v2))
                    v2     = np.subtract(np.array(wp[i+1][0:2]), np.array(stp_sign_f[2:4]))
                    flag_4 = np.sign(np.cross(v1, v2))

                    # Check for line segments intersect.
                    if (flag_1 != flag_2) and (flag_3 != flag_4):
                        coll = True

                    # Check for collinearity.
                    if (flag_1 == 0) and pts_On_Segment(np.array(wp[i][0:2]), np.array(stp_sign_f[0:2]), np.array(wp[i+1][0:2])):
                        coll = True
                    if (flag_2 == 0) and pts_On_Segment(np.array(wp[i][0:2]), np.array(stp_sign_f[2:4]), np.array(wp[i+1][0:2])):
                        coll = True
                    if (flag_3 == 0) and pts_On_Segment(np.array(stp_sign_f[0:2]), np.array(wp[i][0:2]), np.array(stp_sign_f[2:4])):
                        coll = True
                    if (flag_3 == 0) and pts_On_Segment(np.array(stp_sign_f[0:2]), np.array(wp[i+1][0:2]), np.array(stp_sign_f[2:4])):
                        coll = True

                    # Intersection on Road update the goal state to stop before the goal line.
                    if coll:
                        new_idx = i
                        stop_sign_found = True

            # return g_idx, False
            if stop_sign_found is False:
                new_idx = g_idx 

            self._g_idx = new_idx if stop_sign_found else g_idx
            self._goal_state = wp[self._g_idx]

            # If stop sign found, set the goal to zero speed, then transition to 
            # the deceleration state.
            if stop_sign_found:
                self._goal_state[2] = 0
                self._state = 1

        elif self._state == 1:
            self._state = 2 if c_l_v < STOP_THRESHOLD else 1

        # Check to see if we have stayed stopped for at least STOP_COUNTS number of cycles. 
        elif self._state == 2:
            # Post Iteration limit is reached return to lane following.
            if self._stop_count == STOP_COUNTS:
                closest_len, c_idx = get_c_idx(wp, e_s)
                # g_idx = self.get_g_idx(wp, ego_state, closest_len, c_idx)

                arc_length = closest_len
                wp_idx = c_idx
                
                # Closest waypoint is already far enough for the planner
                if arc_length > self._la:
                    return wp_idx

                # End of path
                if wp_idx == len(wp) - 1:
                    return wp_idx

                # Find next waypoint.
                num_wps = len(wp)
                for i in range(wp_idx + 1, num_wps):
                    arc_length += math.sqrt(
                        (wp[i][0] - wp[i-1][0])**2 + (wp[i][1] - wp[i-1][1])**2
                    )
                    if arc_length > self._la:
                        break

                g_idx = i

                # Use the goal index that is the lookahead distance away.

                stop_sign_found = False
                for i in range(c_idx, g_idx):
                    coll = False
                    for stp_sign_f in self._spf:

                        v1     = np.subtract(np.array(wp[i+1][0:2]), np.array(wp[i][0:2]))
                        v2     = np.subtract(np.array(stp_sign_f[0:2]), np.array(wp[i+1][0:2]))
                        flag_1 = np.sign(np.cross(v1, v2))
                        v2     = np.subtract(np.array(stp_sign_f[2:4]), np.array(wp[i+1][0:2]))
                        flag_2 = np.sign(np.cross(v1, v2))

                        v1     = np.subtract(np.array(stp_sign_f[2:4]), np.array(stp_sign_f[0:2]))
                        v2     = np.subtract(np.array(wp[i][0:2]), np.array(stp_sign_f[2:4]))
                        flag_3 = np.sign(np.cross(v1, v2))
                        v2     = np.subtract(np.array(wp[i+1][0:2]), np.array(stp_sign_f[2:4]))
                        flag_4 = np.sign(np.cross(v1, v2))

                        # Check for line segments intersect.
                        if (flag_1 != flag_2) and (flag_3 != flag_4):
                            coll = True

                        # Check for collinearity.
                        if (flag_1 == 0) and pts_On_Segment(np.array(wp[i][0:2]), np.array(stp_sign_f[0:2]), np.array(wp[i+1][0:2])):
                            coll = True
                        if (flag_2 == 0) and pts_On_Segment(np.array(wp[i][0:2]), np.array(stp_sign_f[2:4]), np.array(wp[i+1][0:2])):
                            coll = True
                        if (flag_3 == 0) and pts_On_Segment(np.array(stp_sign_f[0:2]), np.array(wp[i][0:2]), np.array(stp_sign_f[2:4])):
                            coll = True
                        if (flag_3 == 0) and pts_On_Segment(np.array(stp_sign_f[0:2]), np.array(wp[i+1][0:2]), np.array(stp_sign_f[2:4])):
                            coll = True

                        # Intersection on Road update the goal state to stop before the goal line.
                        if coll:
                            # g_idx = i
                            stop_sign_found = True

                # return g_idx, False


                self._g_idx = g_idx
                self._goal_state = wp[g_idx]

                # transition back to our lane following state.
                if not stop_sign_found:
                    self._stop_count = 0
                    self._state = 0

            else:
                self._stop_count += 1

        else:
            raise ValueError('Invalid state value.')
                
    # Velocity Profile
    def foll_check(self, e_s, l_a_pos):
        """Checks for lead vehicle within the proximity of the ego car
        PARAMETERS
        ----------
            1) e_s: ego state vector for the vehicle. (global frame)
                    format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
            2) l_a_pos: The [x, y] position of the lead vehicle.(Global Frame)
        """
        if not self._f_l_a:
            # Euclidean Distance
            dell = [l_a_pos[0] - e_s[0], 
                                     l_a_pos[1] - e_s[1]]
            dist = np.linalg.norm(dell)
            # Out of Bounds
            if dist > self._f_l_a_la:
                return

            dell = np.divide(dell, dist)
            yaw_dell = [math.cos(e_s[2]), 
                                  math.sin(e_s[2])]
            # If lead vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(dell, 
                      yaw_dell) < (1 / math.sqrt(2)):
                return

            self._f_l_a = True

        else:
            dell = [l_a_pos[0] - e_s[0], 
                                     l_a_pos[1] - e_s[1]]
            dist = np.linalg.norm(dell)

            if dist < self._f_l_a_la + 15:
                return
            # If the lead vehicle is still within the ego vehicle's frame of view.
            dell = np.divide(dell, dist)
            yaw_dell = [math.cos(e_s[2]), math.sin(e_s[2])]
            if np.dot(dell, yaw_dell) > (1 / math.sqrt(2)):
                return

            self._f_l_a = False


# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_c_idx(wp, e_s):
    """Gets closest index a given list of waypoints to the vehicle position.
    PARAMETERS
    ----------
        1) wp: current waypoints to track (global frame). 
            format: [[x0, y0, v0]]
        2) e_s: ego state vector for the vehicle. (global frame)
                    format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
    """
    closest_len = float('Inf')
    c_idx = 0
    for i, wp in enumerate(wp):
        d = math.sqrt((e_s[0] - wp[0])**2 + (e_s[1] - wp[1])**2)
        if d < closest_len:
            closest_len = d
            c_idx = i

    return closest_len, c_idx
     
def pts_On_Segment(p1, p2, p3):
    """
    Checks if p1, p2, p3 are collinear.  
    """
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False