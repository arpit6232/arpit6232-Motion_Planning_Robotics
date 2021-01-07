# Configuration Space and Kinematic Analysis 

## Run Commands 
1) To execute Configuration space Analysis Code (Q8) <br/>
    - python python Cspace_2_link.py <br/>
     - Exepects User to Input Link Lengths for 3 links <br/>
     - An Input is required to enter Number of Obstacles, Number of Vertexes per Obstacle, X, Y Coordinate of each obstacle <br/>

2) To Execute Kinematic Analysis, forward and inverse code (Q7) <br/>
    - python 3linkrobot.py <br/>
     - Expects User to Input the Lenght of 3 links
     - Expects user to choose between to input "angle" -> For Forward Kinematic Analsyis 
     - Other option is to type "goal" to Generate a plot to showcase Inverse Kinematic Simulation to the requested Goal 
     - Since It is developed using Newton Rapshon Method of Analysis, initally the code an print a solution to be Unreachable but upon further convergence updates, a close solution of Joint Angles of each Revolute Joint is printed on the screen. 

3) To Execute Code for Minkowski Sum (Q5)
    - python Minkowski_CSpace.py
    - Expects the User to input the number of levels to see in 0 to 360 degrees for the 3d plot