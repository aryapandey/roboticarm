ABOUT THE PROJECT
In this project we have coded a 6 degree of freedom robotic arm to move along :
	two given points (motion control)
	any specified trajectory given its parametric time dependent equation (speed control)
For the above purpose we used forward Kinematics and Inverse Kinematics of a manipulator .
•	Forward Kinematics is used to find the coordinates of the end effector position given joint angles as input.
•	Inverse Kinematics is used to find the joint angles or the angles of the servo motor given coordinates of the end-effector position as input.
The above was done using the Denavit Hartenberg method.
We have also used the Jacobian Matrix in the speed control part to find the speed of the joint angles given the speed of the end effector as input.
The entire code for the manipulator  is written in python language and is stimulated using matplotlib and Axes 3d.
