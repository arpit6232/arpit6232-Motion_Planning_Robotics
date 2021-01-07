import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
import numpy as np
from math import pi
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from workspaces import config


WORKSPACE_CONFIG = config()

show_animation = True


def trajectory_rollout_dynamic_windowing_algo(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, traj = calc_control_and_traj(x, dw, config, goal, ob)

    return u, traj


class RobotType(Enum):
    circle = 0
    rectangle = 1


class Config:
    """
    simulation parameter class
    """

    def __init__(self, ws = None):
        # robot parameter
        if ws == "WO3":
            self.max_speed = 1.0  # [m/s]
            self.min_speed = -0.5  # [m/s]
            self.max_yaw_rate = 90.0 * math.pi / 180.0  # [rad/s]
            self.max_accel = 0.2  # [m/ss]
            self.max_delta_yaw_rate = 90.0 * math.pi / 180.0  # [rad/ss]
            self.v_resolution = 0.01  # [m/s]
            self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
            self.deltaT = 0.1  # [s] Time tick for motion prediction
            self.predict_time = 3.0  # [s]
            self.goal_potential_gain = 0.2
            self.velocity_potential_gain = 1.0
            self.obstacle_cost_gain = 0.5
            self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
            self.robot_type = RobotType.rectangle
        elif ws == "WO2":
            self.max_speed = 1.0  # [m/s]
            self.min_speed = -0.5  # [m/s]
            self.max_yaw_rate = 120.0 * math.pi / 180.0  # [rad/s]
            self.max_accel = 0.2  # [m/ss]
            self.max_delta_yaw_rate = 120.0 * math.pi / 180.0  # [rad/ss]
            self.v_resolution = 0.01  # [m/s]
            self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
            self.deltaT = 0.1  # [s] Time tick for motion prediction
            self.predict_time = 3.0  # [s]
            self.goal_potential_gain = 0.15
            self.velocity_potential_gain = 1.0
            self.obstacle_cost_gain = 0.5
            self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
            self.robot_type = RobotType.rectangle
        elif ws == "WO1":
            self.max_speed = 1.0  # [m/s]
            self.min_speed = -0.5  # [m/s]
            self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
            self.max_accel = 0.2  # [m/ss]
            self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
            self.v_resolution = 0.01  # [m/s]
            self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
            self.deltaT = 0.1  # [s] Time tick for motion prediction
            self.predict_time = 3.0  # [s]
            self.goal_potential_gain = 0.08
            self.velocity_potential_gain = 1.0
            self.obstacle_cost_gain =0.8
            self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
            self.robot_type = RobotType.rectangle

        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.rw = 0.5  # [m] for collision check
        self.rl = 0.75  # [m] for collision check

        self.obstcls = WORKSPACE_CONFIG[ws]
        points = []
        for obs in self.obstcls:
            limit_x, limit_y = obs.exterior.coords.xy
            x_min = np.min(limit_x)
            x_max = np.max(limit_x)
            y_min = np.min(limit_y)
            y_max = np.max(limit_y)

            x_val = np.arange(x_min, x_max+1, 1)
            y_val = np.arange(y_min, y_max+1, 1)

            X, Y = np.meshgrid(x_val,y_val)
            X = X.reshape((np.prod(X.shape),))
            Y = Y.reshape((np.prod(Y.shape),))
            # coords = zip(X,Y)
            for val_x, val_y in zip(X,Y):
                points.append((val_x, val_y))

        self.ob = np.asarray(points)

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value

config = None


def motion(x, u, deltaT):
    """
    motion model
    """

    x[2] += u[1] * deltaT
    x[0] += u[0] * math.cos(x[2]) * deltaT
    x[1] += u[0] * math.sin(x[2]) * deltaT
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    pose_v = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    pose_m = [x[3] - config.max_accel * config.deltaT,
          x[3] + config.max_accel * config.deltaT,
          x[4] - config.max_delta_yaw_rate * config.deltaT,
          x[4] + config.max_delta_yaw_rate * config.deltaT]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(pose_v[0], pose_m[0]), min(pose_v[1], pose_m[1]),
          max(pose_v[2], pose_m[2]), min(pose_v[3], pose_m[3])]

    return dw


def predict_traj(x_init, v, y, config):
    """
    predict traj with an input
    """

    x = np.array(x_init)
    traj = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.deltaT)
        traj = np.vstack((traj, x))
        time += config.deltaT

    return traj


def calc_control_and_traj(x, dw, config, goal, ob):
    """
    Function to sample control and pose of the robot given the constraints
    PARAMETERS
    ----------
        x: X coordinate of the robotss
        dw: Dynamic Window Object
        config: Configuration currently setup for the robot
        goal: Goal of the workspace
        ob: List of obstacles in the workspace
    RETURNS
    -------
        best_u: Sampled Contro 
        best_traj: List of possible trajectory 
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_traj = np.array([x])

    # evaluate all traj with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            traj = predict_traj(x_init, v, y, config)
            # calc cost
            goal_potential = config.goal_potential_gain * calc_to_goal_cost(traj, goal)
            velocity_potential = config.velocity_potential_gain * (config.max_speed - traj[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(traj, ob, config)

            total_cost = goal_potential + velocity_potential + ob_cost

            # search minimum traj
            if min_cost >= total_cost:
                min_cost = total_cost
                best_u = [v, y]
                best_traj = traj
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    # to ensure the robot do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_traj


def calc_obstacle_cost(traj, ob, config):
    """
    Cost function to calculate the obstacle cost inf: collision
    PARAMETERS
    ----------
        traj: list of possible trajectory 
        ob: List of the Obstacles in the workspace
        config: Configuration currently setup for the robot
    RETURNS
    -------
        cost: Cost to Obstacle
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = traj[:, 0] - ox[:, None]
    dy = traj[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = traj[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        _ob = ob[:, None] - traj[:, 0:2]
        _ob = _ob.reshape(-1, _ob.shape[-1])
        _ob = np.array([_ob @ x for x in rot])
        _ob = _ob.reshape(-1, _ob.shape[-1])
        car_top = _ob[:, 0] <= config.rl / 2
        car_right = _ob[:, 1] <= config.rw / 2
        car_bottom = _ob[:, 0] >= -config.rl / 2
        car_left = _ob[:, 1] >= -config.rw / 2
        if (np.logical_and(np.logical_and(car_top, car_right),
                           np.logical_and(car_bottom, car_left))).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK


def calc_to_goal_cost(traj, goal):
    """
    Cost function to calculate the cost to Goal 
    PARAMETERS
    ----------
        traj: list of possible trajectory 
        goal: [gx, gy] Goal Pose
    RETURNS
    -------
        cost: Cost to Goal 
    """

    dx = goal[0] - traj[-1, 0]
    dy = goal[1] - traj[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - traj[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  
    """
    Plots the direction currently pointed by the head of the robot 
    PARAMETERS
    ----------
        x: X location of the arrow (Center of Mass of the robot)
        y: Y Location of the arraw (Center of Mass of the robot)
        length: Length of the robot 
        width: Width of the robot
    RETURNS 
    -------
        None
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  
    """
    Plots the direction currently pointed by the head of the robot 
    PARAMETERS
    ----------
        x: X location of the arrow (Center of Mass of the robot)
        y: Y Location of the arraw (Center of Mass of the robot)
        yaw: Yaw of the robot 
        config: Configuration Setup 
    RETURNS 
    -------
        None
    """
    if config.robot_type == RobotType.rectangle:
        car_dim = np.array([[-config.rl / 2, config.rl / 2,
                             (config.rl / 2), -config.rl / 2,
                             -config.rl / 2],
                            [config.rw / 2, config.rw / 2,
                             - config.rw / 2, -config.rw / 2,
                             config.rw / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        car_dim = (car_dim.T.dot(Rot1)).T
        car_dim[0, :] += x
        car_dim[1, :] += y
        plt.plot(np.array(car_dim[0, :]).flatten(),
                 np.array(car_dim[1, :]).flatten(), "-k")


def main():
    global config

    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])

    ws = input("Enter 1 for Workspace 1, Enter 2 for Workspace 2, Enter 3 for Workspace 3: ")
    ws = int(ws) 
    if(ws == 1):
        ws = "WO1"
        gx = 7.0
        gy = 9.0
    elif (ws == 2):
        ws = "WO2"
        gx = 35.0
        gy = 0.0
    elif (ws == 3):
        ws = "WO3"
        gx = 10.0
        gy = 0.0

    config = Config(ws=ws)

    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # Car is assumed here to be a rectangle
    config.robot_type = RobotType.rectangle

    traj = np.array(x)
    ob = config.ob
    while True:
        u, predicted_traj = trajectory_rollout_dynamic_windowing_algo(x, config, goal, ob)
        x = motion(x, u, config.deltaT)  # simulate robot
        traj = np.vstack((traj, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_traj[:, 0], predicted_traj[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(traj[:, 0], traj[:, 1], "-r")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main()