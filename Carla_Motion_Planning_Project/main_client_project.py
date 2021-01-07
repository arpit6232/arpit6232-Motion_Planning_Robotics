#!/usr/bin/env python3

"""
Code developed from the documentation of 
1) https://carla.readthedocs.io/en/latest/
2) Developed over code initially developed over by Ryan De Iaco, Carlos Wang for Carla Simulation Engine
3) Behavioral Planner, Local Planner, Collision Checker and velocity planner 
    are part of Motion Planning interacting with this Script.
4) This project was developed over Concepts explained at : https://www.coursera.org/specializations/self-driving-cars

CARLA waypoint follower client script.
A controller script to follow a given trajectory, where the trajectory
can be defined using way-points.
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
import controller2d
import configparser 
import local_planner
import behavioural_planner

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv   # Custom live plotting library
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.controller import utils

from timer import Timer

"""
Hyper Parameters
"""

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

WAYPOINTS_FILENAME = 'waypoints.txt'  # waypoint file to load     

C4_STOP_SIGN_FILE        = 'stop_sign_params.txt'
C4_PARKED_CAR_FILE       = 'parked_vehicle_params.txt'


# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/controller_output/'

def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()
    
    # Removing Predistrians for Brevity
    get_non_player_agents_info = False
    if (2 > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info, 
        NumberOfVehicles=2,
        NumberOfPedestrians=0,
        SeedVehicles=0,
        SeedPedestrians=0,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings

# def get_current_pose(measurement):
#     """Pose Estimation Client 
#     """
#     x   = measurement.player_measurements.transform.location.x
#     y   = measurement.player_measurements.transform.location.y
#     yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

#     return (x, y, yaw)

def get_start_pos(scene):
    """Obtains player start x,y, yaw pose from the scene
    """
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

# def get_player_collided_flag(measurement, 
#                              prev_collision_vehicles, 
#                              prev_collision_pedestrians,
#                              prev_collision_other):
#     """Obtains collision flag from player (vehicles, pedestrians, others) """
#     player_meas = measurement.player_measurements
#     curr_coll_vehicles = player_meas.collision_vehicles
#     curr_coll_pedestrians = player_meas.collision_pedestrians
#     curr_coll_other = player_meas.collision_other

#     collided_vehicles = curr_coll_vehicles > prev_collision_vehicles
#     collided_pedestrians = curr_coll_pedestrians > \
#                            prev_collision_pedestrians
#     collided_other = curr_coll_other > prev_collision_other

#     return (collided_vehicles or collided_pedestrians or collided_other,
#             curr_coll_vehicles,
#             curr_coll_pedestrians,
#             curr_coll_other)

def send_control_command(client, throttle, steer, brake, 
                         hand_brake=False, reverse=False):
    """Send control command to CARLA client. """
    control = VehicleControl()
    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0) # Restricted to [-1, 1]
    throttle = np.fmax(np.fmin(throttle, 1.0), 0) # Restricted to [0, 1]
    brake = np.fmax(np.fmin(brake, 1.0), 0) # Restricted to [0, 1]

    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)

# def create_controller_output_dir(output_folder):
#     """
#     Create a Directory in OS(Linux) 
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

# def store_trajectory_plot(graph, fname):
#     """ Store the resulting plot."""
#     create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

#     file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
#     graph.savefig(file_name)

# def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
#     """
#     Create a CSV file for trajectory Generated.
#     """
#     create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
#     file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

#     with open(file_name, 'w') as trajectory_file: 
#         for i in range(len(x_list)):
#             trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
#                                   (x_list[i], y_list[i], v_list[i], t_list[i],
#                                    collided_list[i]))

# def write_collisioncount_file(collided_list):
#     """
#     Create a CSV file for Collision Generated. 
#     """
#     create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
#     file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

#     with open(file_name, 'w') as collision_file: 
#         collision_file.write(str(sum(collided_list)))

def exec_waypoint_nav_demo(args):
    """ 
    Executes waypoint navigation demo.

    Following Snippet of Code was inspired from: https://www.utoronto.ca/news/tags/autonomous-vehicles
    """

    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')

        settings = make_carla_settings(args)
        scene = client.load_settings(settings)
        player_start = 1
        client.start_episode(player_start)
        time.sleep(3)

        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        # live_plotting and live_plotting_period, which controls whether
        # live plotting is enabled or how often the live plotter updates
        # during the simulation run.
        config = configparser.ConfigParser()
        config.read(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))         
        demo_opt = config['Demo Parameters']

        # Get options
        enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))

        # Set options
        live_plot_timer = Timer(live_plot_period)
        
        stop_dat = None
        stp_sign_fs = []     # [x0, y0, x1, y1]
        with open(C4_STOP_SIGN_FILE, 'r') as stopsign_file:
            next(stopsign_file)  # skip header
            stopsign_reader = csv.reader(stopsign_file, 
                                         delimiter=',', 
                                         quoting=csv.QUOTE_NONNUMERIC)
            stop_dat = list(stopsign_reader)
            for i in range(len(stop_dat)):
                stop_dat[i][3] = stop_dat[i][3] * np.pi / 180.0 

        for i in range(len(stop_dat)):
            x = stop_dat[i][0]
            y = stop_dat[i][1]
            z = stop_dat[i][2]
            yaw = stop_dat[i][3] + np.pi / 2.0  
            spos = np.array([
                    [0, 0],
                    [0, 5]]) # 5 - Fence Length Sim Param
            rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
            spos_shift = np.array([
                    [x, x],
                    [y, y]])
            spos = np.add(np.matmul(rotyaw, spos), spos_shift)
            stp_sign_fs.append([spos[0,0], spos[1,0], spos[0,1], spos[1,1]])

        # Parked car(s) Pose, For Static Obstacle Detection
        parkedcar_data = None
        parkedcar_box_pts = []      # [x,y]
        with open(C4_PARKED_CAR_FILE, 'r') as parkedcar_file:
            next(parkedcar_file)  
            parkedcar_reader = csv.reader(parkedcar_file, 
                                          delimiter=',', 
                                          quoting=csv.QUOTE_NONNUMERIC)
            parkedcar_data = list(parkedcar_reader)
            for i in range(len(parkedcar_data)):
                parkedcar_data[i][3] = parkedcar_data[i][3] * np.pi / 180.0 

        # Parked Card Details
        for i in range(len(parkedcar_data)):
            x = parkedcar_data[i][0]
            y = parkedcar_data[i][1]
            z = parkedcar_data[i][2]
            yaw = parkedcar_data[i][3]
            xrad = parkedcar_data[i][4]
            yrad = parkedcar_data[i][5]
            zrad = parkedcar_data[i][6]
            cpos = np.array([
                    [-xrad, -xrad, -xrad, 0,    xrad, xrad, xrad,  0    ],
                    [-yrad, 0,     yrad,  yrad, yrad, 0,    -yrad, -yrad]])
            rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
            cpos_shift = np.array([
                    [x, x, x, x, x, x, x, x],
                    [y, y, y, y, y, y, y, y]])
            cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
            for j in range(cpos.shape[1]):
                parkedcar_box_pts.append([cpos[0,j], cpos[1,j]])

        # Load Waypoints for Parked Car
        waypoints_file = WAYPOINTS_FILENAME
        waypoints_np   = None
        with open(waypoints_file) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle, 
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            waypoints_np = np.array(waypoints)

        controller = controller2d.Controller2D(waypoints)

        # Dynamic Simulation Delta Time
        num_itrs = 10
        if (num_itrs < 1):
            num_itrs = 1

        # Gather current data from the CARLA server. Ego Car Pose and other 
        # details with timestamp 
        measurement_data, sensor_data = client.read_data()
        sim_start_stamp = measurement_data.game_timestamp / 1000.0
        # Send a control command to proceed to next iteration.
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        sim_duration = 0
        # Gather current data on every iteration
        for i in range(num_itrs):
            
            measurement_data, sensor_data = client.read_data()
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            if i == num_itrs - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 -\
                               sim_start_stamp  

        SIMULATION_TIME_STEP = sim_duration / float(num_itrs)
        print("SERVER SIMULATION STEP APPROXIMATION: " + \
              str(SIMULATION_TIME_STEP))
        TOTAL_EPISODE_FRAMES = int((100.00 + 1.00) /\
                               SIMULATION_TIME_STEP) + 300

        # Frame wise Store pose history starting from the start position
        measurement_data, sensor_data = client.read_data()
        start_timestamp = measurement_data.game_timestamp / 1000.0
        
        start_x   = measurement_data.player_measurements.transform.location.x
        start_y   = measurement_data.player_measurements.transform.location.y
        start_yaw = math.radians(measurement_data.player_measurements.transform.rotation.yaw)
        
        
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        x_history     = [start_x]
        y_history     = [start_y]
        yaw_history   = [start_yaw]
        time_history  = [0]
        speed_history = [0]
        collided_flag_history = [False]  # Initialization State

        # Vehicle Trajectory Live Plotting Setup
        # Tested on Linux, can cause system to crash
        lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
        lp_1d = lv.LivePlotter(tk_title="Controls Feedback")
        
        # Add 2D position / trajectory plot
        trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(8, 8), #Inches
                edgecolor="black",
                rect=[0.1, 0.1, 0.8, 0.8])

        trajectory_fig.set_invert_x_axis() 
        trajectory_fig.set_axis_equal()    

        # Waypoint markers
        trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                 x0=waypoints_np[:,0], y0=waypoints_np[:,1],
                                 linestyle="-", marker="", color='g')
        # Trajectory markers
        trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                 x0=[start_x]*TOTAL_EPISODE_FRAMES, 
                                 y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                 color=[1, 0.5, 0])
        # Start and end position marker
        trajectory_fig.add_graph("start_pos", window_size=1, 
                                 x0=[start_x], y0=[start_y],
                                 marker=11, color=[1, 0.5, 0], 
                                 markertext="Start", marker_text_offset=1)

        trajectory_fig.add_graph("end_pos", window_size=1, 
                                 x0=[waypoints_np[-1, 0]], 
                                 y0=[waypoints_np[-1, 1]],
                                 marker="D", color='r', 
                                 markertext="End", marker_text_offset=1)
        # Car marker
        trajectory_fig.add_graph("car", window_size=1, 
                                 marker="s", color='b', markertext="Car",
                                 marker_text_offset=1)
        # Lead car information
        trajectory_fig.add_graph("leadcar", window_size=1, 
                                 marker="s", color='g', markertext="Lead Car",
                                 marker_text_offset=1)
        # Stop sign position
        trajectory_fig.add_graph("stopsign", window_size=1,
                                 x0=[stp_sign_fs[0][0]], y0=[stp_sign_fs[0][1]],
                                 marker="H", color="r",
                                 markertext="Stop Sign", marker_text_offset=1)
        # Stop sign "stop line"
        trajectory_fig.add_graph("stopsign_fence", window_size=1,
                                 x0=[stp_sign_fs[0][0], stp_sign_fs[0][2]],
                                 y0=[stp_sign_fs[0][1], stp_sign_fs[0][3]],
                                 color="r")

        # Parked car points
        parkedcar_box_pts_np = np.array(parkedcar_box_pts)
        trajectory_fig.add_graph("parkedcar_pts", window_size=parkedcar_box_pts_np.shape[0],
                                 x0=parkedcar_box_pts_np[:,0], y0=parkedcar_box_pts_np[:,1],
                                 linestyle="", marker="+", color='b')

        # Lookahead path
        trajectory_fig.add_graph("selected_path", 
                                 window_size=10,
                                 x0=[start_x]*10, 
                                 y0=[start_y]*10,
                                 color=[1, 0.5, 0.0],
                                 linewidth=3)

        # Local path proposals
        for i in range(7):
            trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                     x0=None, y0=None, color=[0.0, 0.0, 1.0])

        # # 1D speed profile updater
        # forward_speed_fig =\
        #         lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
        # forward_speed_fig.add_graph("forward_speed", 
        #                             label="forward_speed", 
        #                             window_size=TOTAL_EPISODE_FRAMES)
        # forward_speed_fig.add_graph("reference_signal", 
        #                             label="reference_Signal", 
        #                             window_size=TOTAL_EPISODE_FRAMES)

        # # Throttle signals graph
        # throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
        # throttle_fig.add_graph("throttle", 
        #                       label="throttle", 
        #                       window_size=TOTAL_EPISODE_FRAMES)

        # # Brake signals graph
        # brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
        # brake_fig.add_graph("brake", 
        #                       label="brake", 
        #                       window_size=TOTAL_EPISODE_FRAMES)

        # # Steering signals graph
        # steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
        # steer_fig.add_graph("steer", 
        #                       label="steer", 
        #                       window_size=TOTAL_EPISODE_FRAMES)

        # Disable Live Plotter
        if not enable_live_plot:
            lp_traj._root.withdraw()
            lp_1d._root.withdraw()        


        # Local Planner Variables
        wp_goal_idx   = 0
        local_waypoints = None
        path_validity   = np.zeros((7, 1), dtype=bool)
        lp = local_planner.LocalPlanner(7, # Number of Paths
                                        1.5, # Path Offset
                                        [-1.0, 1.0, 3.0], # Circle Offsets
                                        [1.5, 1.5, 1.5], # Circle Radii
                                        10, # Path Select Weight
                                        1.0, # Time Gap
                                        1.5, # Max Acceleration Bound
                                        2.0, # Velocity Bounds
                                        3.5) # Stop Line Buffer

        bp = behavioural_planner.BehaviouralPlanner(8.0,
                                                    stp_sign_fs,
                                                    20.0) # Lead Vehicle Limit in Meters

        # Scenario Execution Loop

        # Iterate the frames until the end of the waypoints is reached or
        # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
        # ouptuts the results to the controller output directory.
        reached_the_end = False
        skip_first_frame = True

        current_timestamp = start_timestamp
        prev_collision_vehicles    = 0
        prev_collision_pedestrians = 0
        prev_collision_other       = 0

        for frame in range(TOTAL_EPISODE_FRAMES):
            measurement_data, sensor_data = client.read_data()
            prev_timestamp = current_timestamp


            # curr_x, curr_y, curr_yaw = get_current_pose(measurement_data)
            curr_x   = measurement_data.player_measurements.transform.location.x
            curr_y   = measurement_data.player_measurements.transform.location.y
            curr_yaw = math.radians(measurement_data.player_measurements.transform.rotation.yaw)
            
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(measurement_data.game_timestamp) / 1000.0

            # Begin Control Communication
            if current_timestamp <= 1.00:
                send_control_command(client, throttle=0.0, steer=0, brake=1.0)
                continue
            else:
                current_timestamp = current_timestamp - 1.00
            
            x_history.append(curr_x)
            y_history.append(curr_y)
            yaw_history.append(curr_yaw)
            speed_history.append(current_speed)
            time_history.append(current_timestamp) 

            player_meas = measurement_data.player_measurements
            curr_coll_vehicles = player_meas.collision_vehicles
            curr_coll_pedestrians = player_meas.collision_pedestrians
            curr_coll_other = player_meas.collision_other

            collided_vehicles = curr_coll_vehicles > prev_collision_vehicles
            collided_pedestrians = curr_coll_pedestrians > \
                                prev_collision_pedestrians
            collided_other = curr_coll_other > prev_collision_other

            collided_flag = collided_vehicles or collided_pedestrians or collided_other
            prev_collision_vehicles = curr_coll_vehicles
            curr_coll_pedestrians = curr_coll_pedestrians,
            prev_collision_other = curr_coll_other
        
            collided_flag_history.append(collided_flag)

            # Local Planner Update:
            #   This will use the behavioural_planner.py and local_planner.py
            #   implementations 

            # Obtain Lead Vehicle information.
            lead_car_pos    = []
            lead_car_length = []
            lead_car_speed  = []
            for agent in measurement_data.non_player_agents:
                agent_id = agent.id
                if agent.HasField('vehicle'):
                    lead_car_pos.append(
                            [agent.vehicle.transform.location.x,
                             agent.vehicle.transform.location.y])
                    lead_car_length.append(agent.vehicle.bounding_box.extent.x)
                    lead_car_speed.append(agent.vehicle.forward_speed)

            # Execute the behaviour and local planning in the current instance 
            # Frame Wise Update of Planner Input
            if frame % 2 == 0:
                open_loop_speed = lp._velocity_planner.feedforward_speed(current_timestamp - prev_timestamp)
                ego_state = [curr_x, curr_y, curr_yaw, open_loop_speed]
                bp.set_lookahead(8.0 + 2.0 * open_loop_speed)
                bp.trans_state(waypoints, ego_state, current_speed)
                bp.foll_check(ego_state, lead_car_pos[1])
                goal_state_set = lp.possible_goal_set(bp._g_idx, bp._goal_state, waypoints, ego_state)
                paths, path_validity = lp.trajectory_plan(goal_state_set)
                paths = local_planner.frame_adj(paths, ego_state)
                collision_check_array = lp._collision_checker.collision_check(paths, [parkedcar_box_pts])
                best_idx = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                if best_idx == None:
                    best_path = lp._prev_best_path
                else:
                    best_path = paths[best_idx]
                    lp._prev_best_path = best_path

                # Velocity profile for the path, and compute waypoints
                desired_speed = bp._goal_state[2]
                lead_c_s = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
                decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
                local_waypoints = lp._velocity_planner.calc_vel_profile(best_path, \
                    desired_speed, ego_state, current_speed, decelerate_to_stop, lead_c_s, bp._f_l_a)

                if local_waypoints != None:
                    wp_distance = []   # distance array
                    local_waypoints_np = np.array(local_waypoints)
                    for i in range(1, local_waypoints_np.shape[0]):
                        wp_distance.append(np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i-1, 0])**2 +
                                        (local_waypoints_np[i, 1] - local_waypoints_np[i-1, 1])**2))

                    wp_distance.append(0)  
                    wp_interp      = []    
                    for i in range(local_waypoints_np.shape[0] - 1):
                        wp_interp.append(list(local_waypoints_np[i]))
                
                        num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                                     float(0.01)) - 1)
                        wp_vector = local_waypoints_np[i+1] - local_waypoints_np[i]
                        wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                        for j in range(num_pts_to_interp):
                            next_wp_vector = 0.01 * float(j+1) * wp_uvector
                            wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))

                    wp_interp.append(list(local_waypoints_np[-1]))
                    controller.update_waypoints(wp_interp)
                    pass

            # Controller Update
            if local_waypoints != None and local_waypoints != []:
                controller.update_values(curr_x, curr_y, curr_yaw, 
                                         current_speed,
                                         current_timestamp, frame)
                controller.update_controls()
                cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            else:
                cmd_throttle = 0.0
                cmd_steer = 0.0
                cmd_brake = 0.0

            if skip_first_frame and frame == 0:
                pass
            elif local_waypoints == None:
                pass
            else:
                # Live plotter
                trajectory_fig.roll("trajectory", curr_x, curr_y)
                trajectory_fig.roll("car", curr_x, curr_y)
                if lead_car_pos:    
                    trajectory_fig.roll("leadcar", lead_car_pos[1][0],
                                        lead_car_pos[1][1])
                # forward_speed_fig.roll("forward_speed", 
                #                        current_timestamp, 
                #                        current_speed)
                # forward_speed_fig.roll("reference_signal", 
                #                        current_timestamp, 
                #                        controller._desired_speed)
                # throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
                # brake_fig.roll("brake", current_timestamp, cmd_brake)
                # steer_fig.roll("steer", current_timestamp, cmd_steer)

                # Local path plotter update
                if frame % 2 == 0:
                    path_counter = 0
                    for i in range(7):
                        if path_validity[i]:
                            # Colour paths according to collision checking.
                            if not collision_check_array[path_counter]:
                                colour = 'r'
                            elif i == best_idx:
                                colour = 'k'
                            else:
                                colour = 'b'
                            trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                            path_counter += 1
                        else:
                            trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
               
                wp_interp_np = np.array(wp_interp)
                path_indices = np.floor(np.linspace(0, 
                                                    wp_interp_np.shape[0]-1,
                                                    10))
                trajectory_fig.update("selected_path", 
                        wp_interp_np[path_indices.astype(int), 0],
                        wp_interp_np[path_indices.astype(int), 1],
                        new_colour=[1, 0.5, 0.0])

                if enable_live_plot and \
                   live_plot_timer.has_exceeded_lap_period():
                    lp_traj.refresh()
                    lp_1d.refresh()
                    live_plot_timer.lap()

            # Output controller command to CARLA server
            send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

            # If end of waypoint.
            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - curr_x,
                waypoints[-1][1] - curr_y]))
            if  dist_to_last_waypoint < 2.0:
                reached_the_end = True
            if reached_the_end:
                break

        if reached_the_end:
            print("Reached the end of path. Writing to controller_output...")
        else:
            print("Exceeded assessment time. Writing to controller_output...")
        # Stop the car
        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
        # # # Store the variables
        store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
        # store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
        # store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
        # store_trajectory_plot(brake_fig.fig, 'brake_output.png')
        # store_trajectory_plot(steer_fig.fig, 'steer_output.png')
        # write_trajectory_file(x_history, y_history, speed_history, time_history,
        #                       collided_flag_history)
        # write_collisioncount_file(collided_flag_history)

def main():
    """Main function.
    PARAMETERS
    ----------
        -v, --verbose: print debug information
        --host: IP of the host server (default: localhost)
        -p, --port: TCP port to listen to (default: 2000)
        -a, --autopilot: enable autopilot
        -q, --quality-level: graphics quality level [Low or Epic]
        -i, --images-to-disk: save images to disk
        -c, --carla-settings: Path to CarlaSettings.ini file
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    while True:
        try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')