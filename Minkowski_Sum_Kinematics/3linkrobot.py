import numpy as np 
import matplotlib.pyplot as plt 
import math
import enum

"""

Kinematic Analysis of 3 link robot is undertaken using Newton Raphson method of Inverse Jacobian 

The following, Code excerpt was read and followed from the book " Modern Robotics - Motion Planning and Control "

 Assume thatf:Rn→Rm is differentiable, and letxdbe the desired end-effector coordinates.  
 Theeta(θ)for the Newton-Raphson method is defined as g(θ) = xd−f(θ), and the goal is to find joint coordinates θd such that
 g(θd) = xd − f(θd) = 0.

 Given an initial guessθ0 which is “close to” a solution θd, the kinematics can be expressed as the Taylor expansion. 
 Ignoring the Higher Order Terms, The Jacobian forms, Update Step leads to convergence. 

 The first step of the Newton–Raphson method for nonlinear root-finding for  a  scalar x and θ.   
 In  the  first  step,  the  slope −∂f/∂θ is  evaluated  at  the  point(θ0,xd−f(θ0)).  
 In the second step, the slope is evaluated at the point (θ1,xd−f(θ1)) and eventually the process converges to θd.
 Note that an initial guess to the left of the plateau of xd−f(θ) would be likely to result in convergence to the other root of 
 xd−f(θ), and an initial guess at or near the plateau would result in a large initial |∆θ| and the iterative process might not 
 converge at all.

 PseudoInverse needs to be used, since this is not a square matrix. 
 Replacing the Jacobian inverse with the pseudoinverse, ∆θ = J†(θ0) (xd−f(θ0))

 The Following Pseudo Code explains the overview, 
 Newton–Raphson iterative algorithm for finding θd: 
 (a) Initialization:  Given xd∈Rm and an initial guess θ0∈Rn, set 
    i= 0. 
 (b) Set e=xd−f(θi).  While epsilon for some small theta : Set θi+1=θi+J†(θi)e.
 ->Incrementi

"""

class State_Machine(enum.Enum):
   UPDATE_GOAL = 1
   MOVE = 2

class Robot(object):
    """
    This Class is helper class for plotting and mainipulating which keeps track the end points. 
    @param - arm_lengths : Array of the Lengths of each arm 
    @param - motor_angles : Current Angle of the Each Revolute Joint in the Global Frame
    @goal  - Final Position expected to be reached by the link   
    """
    def __init__(self, arm_lengths, motor_angles, goal):
        """
        Initialization 
        """
        self.arm_lengths = np.array(arm_lengths)
        self.motor_angles = np.array(motor_angles)
        self.link_end_pts = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.goal = np.array(goal).T
        self.lim = sum(arm_lengths)
        plt.ion()
        plt.show()
        # Find the Location of End Points of each Link
        for i in range(1, 4):
            self.link_end_pts[i][0] = self.link_end_pts[i-1][0] + self.arm_lengths[i-1] * \
                np.cos(np.sum(self.motor_angles[:i]))
            self.link_end_pts[i][1] = self.link_end_pts[i-1][1] + self.arm_lengths[i-1] * \
                np.sin(np.sum(self.motor_angles[:i]))

        # Explicity Setting The end effector Position
        self.end_effector = np.array(self.link_end_pts[3]).T
        self.plot()

    def update_joints(self, motor_angles):
        """
        Update the Location of the end points of the link, Based on Updates of the End points 
        """
        self.motor_angles = motor_angles

        # Update Steps 
        # Set e=xd−f(θi).  While epsilon for some small theta : Set θi+1=θi+J†(θi)e. 
        for i in range(1, 4):
            # Cosine length Update
            self.link_end_pts[i][0] = self.link_end_pts[i-1][0] + self.arm_lengths[i-1] * \
                np.cos(np.sum(self.motor_angles[:i]))
            # Sine length Update
            self.link_end_pts[i][1] = self.link_end_pts[i-1][1] + self.arm_lengths[i-1] * \
                np.sin(np.sum(self.motor_angles[:i]))

        # Explicity Setting The end effector Position
        self.end_effector = np.array(self.link_end_pts[3]).T
        self.plot()

    def plot(self): 
        """
        Helper functions to plot links, Motor joints, based on Newton Raphson Jacobian Inverse Calculation 
        """
        plt.cla()
        for i in range(4):
            if i is not 3:
                # Plot Links 
                plt.plot([self.link_end_pts[i][0], self.link_end_pts[i+1][0]],\
                     [self.link_end_pts[i][1], self.link_end_pts[i+1][1]], 'c-')
            # Plot Motor Joint 
            plt.plot(self.link_end_pts[i][0], self.link_end_pts[i][1], 'ko')

        # Mark the goal Position
        plt.plot(self.goal[0], self.goal[1], 'rx')
        plt.xlim([-self.lim, self.lim])
        plt.ylim([-self.lim, self.lim])
        plt.draw()
        plt.pause(0.0001)


def inv_K(arm_lengths, motor_angles, goal):
    """
    Inverse Kinematics for Analysis to calculate Jacobian, to update the non-linear equations ignoring the higher order terms
    """
    # Number of Iterations here is 30000, 
    for itr in range(30000): 
        J = np.zeros((2, 3))
        # Calculates the Forward Kineamatics of the robot for the current transform
        transform_t = forw_K(arm_lengths, motor_angles)

        epsilon, distance = np.array([(goal[0] - transform_t[0]), (goal[1] - transform_t[1])]).T,\
             np.hypot((goal[0] - transform_t[0]), (goal[1] - transform_t[1]))
        
        # Success Condition 
        if distance < 1:
            return motor_angles, True

        # Update Jacobian 
        for i in range(3):
            J[0, i] = J[1, i] = 0
            for j in range(i, 3):
                J[0, i] -= arm_lengths[j] * np.sin(np.sum(motor_angles[:j]))
                J[1, i] += arm_lengths[j] * np.cos(np.sum(motor_angles[:j]))

        print("Jacobian")
        print(J)
        # Angle Update Step 
        # θi+1=θi+J†(θi)e. 
        motor_angles = motor_angles + np.dot(np.linalg.pinv(J), epsilon)
        print("Motor Angles")
        print(np.degrees(motor_angles))
    return motor_angles, False


def forw_K(arm_lengths, motor_angles):
    """
    Function to Calculate the forward kinematics.
    """
    pos_x = pos_y = 0
    # Simple logic gets the calculates the End Effector position 
    # from the current motor angle and position
    for i in range(1, 4):
        pos_x += arm_lengths[i-1] * np.cos(np.sum(motor_angles[:i]))
        pos_y += arm_lengths[i-1] * np.sin(np.sum(motor_angles[:i]))

    print("Forward Kinematic Solution")
    print(np.array([pos_x, pos_y]).T)

    # Transpose is necessary, for future updates
    return np.array([pos_x, pos_y]).T


def main():
    """
    Main functionalities 
    """

    # Length Setup 
    l1 = input("Enter Length of Link 1\n")
    l2 = input("Enter Length of Link 2\n")
    l3 = input("Enter Length of Link 3\n")
    arm_lengths = [float(l1), float(l2), float(l3)]
    
    # Default Values 
    motor_angles = np.array([np.radians(30)] * 3)
    goal_pos = [0.1, 4.1]
    output = input("Do you want to Enter angles (Enter 'angle') or Enter final goal(Enter 'goal')")

    # Calculates the final Goal from angle 
    if(output == 'angle'):
        motor_angle1 = input("Enter Angle of Link 1\n")
        motor_angle2 = input("Enter Angle of Link 2\n")
        motor_angle3 = input("Enter Angle of Link 3\n")
        motor_angles = np.radians(np.array([float(motor_angle1), float(motor_angle2), float(motor_angle3)]))
        print("Wanted Motor Angles")
        print(np.degrees(motor_angles))
        final_goal_pos = forw_K(arm_lengths, motor_angles)
        print("End Effector Position")
        print(final_goal_pos)
    
    # For inverse kinematics 
    elif (output == 'goal'):
        goal_x = input("Enter X Coordinate of Goal\n")
        goal_y = input("Enter Y Coordinate of Goal\n")
        final_goal_pos = [float(goal_x), float(goal_y)]

    # Object of concern 
    arm = Robot(arm_lengths, motor_angles, goal_pos)
    # Initializes the State to some default 
    state = State_Machine.UPDATE_GOAL
    solution_found = False

    # Helper Flag 
    goal_counter = 0
    while True:
        # Helper functions 
        old_goal = np.array(goal_pos)
        goal_pos = np.array(arm.goal)
        distance = np.hypot((goal_pos[0] - arm.end_effector[0]), (goal_pos[1] - arm.end_effector[1]))

        # Inspired from the concept of Embedded Systems which uses State Machine 
        # To stay in an infinite connected loop 
        if state is State_Machine.UPDATE_GOAL:
            # Sucess condition 
            if distance > 0.1 and not solution_found:
                # Gives the Updates over convergence 
                joint_goal_angles, solution_found = inv_K(arm_lengths, motor_angles, goal_pos)
                if not solution_found:
                    # Still Convergence Condition is not met 
                    print("Goal Unreachable")
                    state = State_Machine.UPDATE_GOAL
                    arm.goal = final_goal_pos
                elif solution_found:
                    # Continue Updates
                    state = State_Machine.MOVE
                    arm.goal = final_goal_pos
                if distance < 0.1:
                    # Sucess Condition
                    print("Joint Angles")
                    np.radians(np.degrees(motor_angles))
                    break
        # Second State Machine,         
        elif state is State_Machine.MOVE:
            # Motor Angle Updates
            if distance > 0.1 and all(old_goal == goal_pos):
                motor_angles = motor_angles + (2 * ((joint_goal_angles - motor_angles + np.pi) % (2 * np.pi) - np.pi) * 0.01)
            else:
                # Update State Machine 
                state = State_Machine.UPDATE_GOAL
                solution_found = False
                arm.goal = final_goal_pos
                goal_counter += 1

            if distance < 1:
                    print("Joint Angles")
                    print(np.degrees(np.asarray(motor_angles)))

        # Runs 5 iterations to goal counter for success
        if goal_counter >= 5:
            break
        
        # Jacobian Update
        arm.update_joints(motor_angles)


if __name__ == '__main__':
    main()
