#!/usr/bin/env python3

import numpy as np
from math import sin, cos, pi, sqrt

"""
1) Linked Research Paper: M. Mcnaughton, C. Urmson, J. M. Dolan, and J.-W. Lee, “Motion
planning for autonomous driving with a conformal spatiotemporal lat-
tice,” 2011 IEEE International Conference on Robotics and Automation,
2011. This paper introduces the concepts behind generating a conformal
spatiotemporal lattice for on-road motion planning.
"""

class VelocityPlanner:
    def __init__(self, sim_time_delta, a_max, slow_speed, stop_buff):
        self._sim_time_delta         = sim_time_delta
        self._a_max            = a_max
        self._slow_speed       = slow_speed
        self._stop_buff = stop_buff
        self._prev_trej  = [[0.0, 0.0, 0.0]]

    def feedforward_speed(self, dT):
        """
        Computes an open loop speed based on delta t of the simulator
        """
        if len(self._prev_trej) == 1:
            return self._prev_trej[0][2] 
        
        if dT < 1e-4:
            return self._prev_trej[0][2]

        for i in range(len(self._prev_trej)-1):
            distance_step = np.linalg.norm(np.subtract(self._prev_trej[i+1][0:2], 
                                                       self._prev_trej[i][0:2]))
            velocity = self._prev_trej[i][2]
            time_delta = distance_step / velocity
           
            # Incase Delta is too big, interpoloation 
            if time_delta > dT:
                v1 = self._prev_trej[i][2]
                v2 = self._prev_trej[i+1][2]
                return v1 + (dT / time_delta) * (v2 - v1)

            else:
                dT -= time_delta

        # Returns the end velocity of the trajectory.
        return self._prev_trej[-1][2]

    def calc_vel_profile(self, path, d_s, e_s, 
                                 feedback_speed, decelerate_to_stop, 
                                 lead_c_s, follow_lead_vehicle):
        """Computes the velocity profile for the local planner path.
        
        PARAMETERS
        ----------
            1) paths: A list of paths in the global frame.  
                A path is a list of points of the following format:
                    [x_points, y_points, t_points]:
                        t_points: List of yaw values (rad)
            2) d_s: Goal Ego Car Speed
            3) e_s: ego state vector for the vehicle(Global Frame).
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
            4) feedback_speed: current (closed-loop) speed for vehicle (m/s)
            5) decelerate_to_stop: Flag for if to decelerate to stop
            6) lead_c_s: the lead vehicle current state.
                Format: [lead_car_x, lead_car_y, lead_car_speed]
            7) follow_lead_vehicle: Flag to update State based on influence the speed
             profile of the local path.

            # brake_idx : index of the path at which start braking.
            # decel_idx : index at which we stop decelerating to our slow_speed. 
        RETURNS
        -------
            prof: Updated profile which contains the local path as well as
                the speed to be tracked by the controller (global frame).
                Format: [[xm, ym, vm]]
        """
        prof = []
        s_vel = e_s[3]

        # Literature suggests on using trapezoidal profile to decelerate to stop.
        if decelerate_to_stop:
            
            prof = []
            slow_speed = self._slow_speed
            stop_buff = self._stop_buff

            # decel_D : start_speed to some coasting speed (slow_speed).
            # brake_D : slow_speed to 0, both at a constant deceleration.
            decel_D = c_d(s_vel, slow_speed, -self._a_max)
            brake_D = c_d(slow_speed, 0, -self._a_max)

            # Total Path Length
            path_norm = 0.0
            for i in range(len(path[0])-1):
                path_norm += np.linalg.norm([path[0][i+1] - path[0][i], 
                                            path[1][i+1] - path[1][i]])

            stop_idx = len(path[0]) - 1
            temp_dist = 0.0
            # Index to Stop
            while (stop_idx > 0) and (temp_dist < stop_buff):
                temp_dist += np.linalg.norm([path[0][stop_idx] - path[0][stop_idx-1], 
                                            path[1][stop_idx] - path[1][stop_idx-1]])
                stop_idx -= 1

            # Trapeziodal Calculation for Hard deceleration
            if brake_D + decel_D + stop_buff > path_norm:
                speeds = []
                vf = 0.0
                for i in reversed(range(stop_idx, len(path[0]))):
                    speeds.insert(0, 0.0)
                for i in reversed(range(stop_idx)):
                    dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                        path[1][i+1] - path[1][i]])
                    vi = c_f_s(vf, -self._a_max, dist)
                    # Clamping within bounds of velocity
                    if vi > s_vel:
                        vi = s_vel

                    speeds.insert(0, vi)
                    vf = vi

                # Generate Profile
                for i in range(len(speeds)):
                    prof.append([path[0][i], path[1][i], speeds[i]])
                
            # Generating complete trapezoidal profile.
            else:
                brake_idx = stop_idx 
                temp_dist = 0.0
                while (brake_idx > 0) and (temp_dist < brake_D):
                    temp_dist += np.linalg.norm([path[0][brake_idx] - path[0][brake_idx-1], 
                                                path[1][brake_idx] - path[1][brake_idx-1]])
                    brake_idx -= 1

                decel_idx = 0
                temp_dist = 0.0
                while (decel_idx < brake_idx) and (temp_dist < decel_D):
                    temp_dist += np.linalg.norm([path[0][decel_idx+1] - path[0][decel_idx], 
                                                path[1][decel_idx+1] - path[1][decel_idx]])
                    decel_idx += 1

                # Speed to decel_idx should be a linear ramp from current speed down
                # to slow_speed, decelerating at -self._a_max.
                vi = s_vel
                for i in range(decel_idx): 
                    dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                        path[1][i+1] - path[1][i]])
                    vf = c_f_s(vi, -self._a_max, dist)

                    # Clambing for lower bounds on velocity
                    if vf < slow_speed:
                        vf = slow_speed

                    prof.append([path[0][i], path[1][i], vi])
                    vi = vf

                for i in range(decel_idx, brake_idx):
                    prof.append([path[0][i], path[1][i], vi])
                    
                for i in range(brake_idx, stop_idx):
                    dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                        path[1][i+1] - path[1][i]])
                    vf = c_f_s(vi, -self._a_max, dist)
                    prof.append([path[0][i], path[1][i], vi])
                    vi = vf

                # Complete trajectory
                for i in range(stop_idx, len(path[0])):
                    prof.append([path[0][i], path[1][i], 0.0])

        elif follow_lead_vehicle:
            prof = []
            # Find the closest point to the lead vehicle on our planned path.
            min_idx = len(path[0]) - 1
            min_dist = float('Inf')
            for i in range(len(path)):
                dist = np.linalg.norm([path[0][i] - lead_c_s[0], 
                                    path[1][i] - lead_c_s[1]])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i

            d_s = min(lead_c_s[2], d_s)
            r_idx = min_idx
            distance = min_dist
            distance_gap = d_s * self._sim_time_delta
            while (r_idx > 0) and (distance > distance_gap):
                distance += np.linalg.norm([path[0][r_idx] - path[0][r_idx-1], 
                                            path[1][r_idx] - path[1][r_idx-1]])
                r_idx -= 1

            # Maintain Speed within Bounds Distance Calculation
            if d_s < s_vel:
                decel_D = c_d(s_vel, d_s, -self._a_max)
            else:
                decel_D = c_d(s_vel, d_s, self._a_max)

            # Speed Profile
            vi = s_vel
            for i in range(r_idx + 1):
                dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                    path[1][i+1] - path[1][i]])
                if d_s < s_vel:
                    vf = c_f_s(vi, -self._a_max, dist)
                else:
                    vf = c_f_s(vi, self._a_max, dist)

                prof.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(r_idx + 1, len(path[0])):
                prof.append([path[0][i], path[1][i], d_s])

        else:
            prof = []
            # Distance Calculations
            if d_s < s_vel:
                accel_distance = c_d(s_vel, d_s, -self._a_max)
            else:
                accel_distance = c_d(s_vel, d_s, self._a_max)

            # Ramp end of velocity profile.
            r_idx = 0
            distance = 0.0
            while (r_idx < len(path[0])-1) and (distance < accel_distance):
                distance += np.linalg.norm([path[0][r_idx+1] - path[0][r_idx], 
                                            path[1][r_idx+1] - path[1][r_idx]])
                r_idx += 1

            vi = s_vel
            for i in range(r_idx):
                dist = np.linalg.norm([path[0][i+1] - path[0][i], 
                                    path[1][i+1] - path[1][i]])
                if d_s < s_vel:
                    vf = c_f_s(vi, -self._a_max, dist)
                    if vf < d_s:
                        vf = d_s
                else:
                    vf = c_f_s(vi, self._a_max, dist)
                    if vf > d_s:
                        vf = d_s

                prof.append([path[0][i], path[1][i], vi])
                vi = vf

            for i in range(r_idx+1, len(path[0])):
                prof.append([path[0][i], path[1][i], d_s])

        # Interpolation
        if len(prof) > 1:
            i_s = [(prof[1][0] - prof[0][0]) * 0.1 + prof[0][0], 
                                  (prof[1][1] - prof[0][1]) * 0.1 + prof[0][1], 
                                  (prof[1][2] - prof[0][2]) * 0.1 + prof[0][2]]
            del prof[0]
            prof.insert(0, i_s)

        self._prev_trej = prof

        return prof

def c_d(v_i, v_f, a):
    """Computes the distance given an initial and final velocity, with a constant
    acceleration.
    """
    d = (v_f**2 - v_i**2) / (2 * a)
    return d

def c_f_s(v_i, a, d):
    """Computes the final speed given an initial velocity, distance travelled, 
    and a constant acceleration.
    """
    dis = v_i**2 + 2 * a * d
    v_f = sqrt(dis) if dis > 0 else 0
    return v_f