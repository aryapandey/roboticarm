

from sympy import symbols, cos, sin, pi, simplify, pprint, tan, expand_trig, sqrt, trigsimp, atan2
from sympy.matrices import Matrix

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import numpy as np
from inverse import *


# declaring the link length of the arm
a1 = 1.0
a2 = 1.0
a3 = 5.0
a4 = 5.0
a5 = 0.2

dt = 0.001  # choosing dt = 1 ms

# this function prints some latters or words when ginven points
def print_word():
    fig, ax = create_plot()

    # corner points of word
    X0 = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    Y0 = [10,8.5,7,7.75,9.25,7.75,7,6,6,4.5,4.5,6,4.5,3.5,2,2.75,3.5,2.75,3.5,2.5,1,-0.5,0.25,1.75,0.25,-0.5]
    Z0 = [1,4,1,2.5,2.5,2.5,1,1,4,4,3,3,1,1,4,2.5,4,2.5,1,1,4,1,2.5,2.5,2.5,1]

    X, Y, Z, X1, Y1, Z1 = [], [], [], [], [], []
    for i in range(26):
        px, py, pz = X0[i], Y0[i], Z0[i]
        X1 = X0[:i+1]; Y1 = Y0[:i+1]; Z1 = Z0[:i+1]
        X1 = np.reshape(X1, (1, i+1))
        Y1 = np.reshape(Y1, (1, i+1))
        Z1 = np.reshape(Z1, (1, i+1))
        # print(X1, Y1, Z1)

        roll, pitch, yaw = 0, 0, 0
        q1, q2, q3, q4, q5, q6 = get_angles(px, py, pz, roll, pitch, yaw)
        X, Y, Z = forward_kin(q1, q2, q3, q4, q5, q6)
        update_plot(X, Y, Z, X1, Y1, Z1,  fig, ax)

    # print(X1, Y1, Z1)
    # update_plot(X, Y, Z, X1, Y1, X1, fig, ax)
    plt.show()


# this function spit outs the speed of change of angles of joints using jacobian matrix (checkout this link to know more ---> https://youtu.be/SefTCXrpL8U )
def get_speed(vx, vy, vz, valpha, vbeta, vgama, q1, q2, q3, q4, q5, q6):
    # DH parameter table
    DH = np.array([[q1, np.pi / 2, a2, a1],
                   [q2, 0, a3, 0],
                   [q3 + np.pi / 2, np.pi / 2, 0, 0],
                   [q4, -np.pi / 2, 0, a4],
                   [q5, np.pi / 2, 0, 0],
                   [q6, 0, 0, a5]])

    # homogeneous matrices
    H0_0 = Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    H0_1 = Matrix([[np.cos(DH[0, 0]), -np.sin(DH[0, 0]) * np.cos(DH[0, 1]), np.sin(DH[0, 0]) * np.sin(DH[0, 1]),
                    DH[0, 2] * np.cos(DH[0, 0])],
                   [np.sin(DH[0, 0]), np.cos(DH[0, 0]) * np.cos(DH[0, 1]), -np.cos(DH[0, 0]) * np.sin(DH[0, 1]),
                    DH[0, 2] * np.sin(DH[0, 0])],
                   [0, np.sin(DH[0, 1]), np.cos(DH[0, 1]), DH[0, 3]],
                   [0, 0, 0, 1]])

    H1_2 = Matrix([[np.cos(DH[1, 0]), -np.sin(DH[1, 0]) * np.cos(DH[1, 1]), np.sin(DH[1, 0]) * np.sin(DH[1, 1]),
                    DH[1, 2] * np.cos(DH[1, 0])],
                   [np.sin(DH[1, 0]), np.cos(DH[1, 0]) * np.cos(DH[1, 1]), -np.cos(DH[1, 0]) * np.sin(DH[1, 1]),
                    DH[1, 2] * np.sin(DH[1, 0])],
                   [0, np.sin(DH[1, 1]), np.cos(DH[1, 1]), DH[1, 3]],
                   [0, 0, 0, 1]])

    H0_2 = np.dot(H0_1, H1_2)

    H2_3 = Matrix([[np.cos(DH[2, 0]), -np.sin(DH[2, 0]) * np.cos(DH[2, 1]), np.sin(DH[2, 0]) * np.sin(DH[2, 1]),
                    DH[2, 2] * np.cos(DH[2, 0])],
                   [np.sin(DH[2, 0]), np.cos(DH[2, 0]) * np.cos(DH[2, 1]), -np.cos(DH[2, 0]) * np.sin(DH[2, 1]),
                    DH[2, 2] * np.sin(DH[2, 0])],
                   [0, np.sin(DH[2, 1]), np.cos(DH[2, 1]), DH[2, 3]],
                   [0, 0, 0, 1]])

    H0_3 = np.dot(H0_2, H2_3)

    H3_4 = Matrix([[np.cos(DH[3, 0]), -np.sin(DH[3, 0]) * np.cos(DH[3, 1]), np.sin(DH[3, 0]) * np.sin(DH[3, 1]),
                    DH[3, 2] * np.cos(DH[3, 0])],
                   [np.sin(DH[3, 0]), np.cos(DH[3, 0]) * np.cos(DH[3, 1]), -np.cos(DH[3, 0]) * np.sin(DH[3, 1]),
                    DH[3, 2] * np.sin(DH[3, 0])],
                   [0, np.sin(DH[3, 1]), np.cos(DH[3, 1]), DH[3, 3]],
                   [0, 0, 0, 1]])

    H0_4 = np.dot(H0_3, H3_4)

    H4_5 = Matrix([[np.cos(DH[4, 0]), -np.sin(DH[4, 0]) * np.cos(DH[4, 1]), np.sin(DH[4, 0]) * np.sin(DH[4, 1]),
                    DH[4, 2] * np.cos(DH[4, 0])],
                   [np.sin(DH[4, 0]), np.cos(DH[4, 0]) * np.cos(DH[4, 1]), -np.cos(DH[4, 0]) * np.sin(DH[4, 1]),
                    DH[4, 2] * np.sin(DH[4, 0])],
                   [0, np.sin(DH[4, 1]), np.cos(DH[4, 1]), DH[4, 3]],
                   [0, 0, 0, 1]])

    H0_5 = np.dot(H0_4, H4_5)

    H5_6 = Matrix([[np.cos(DH[5, 0]), -np.sin(DH[5, 0]) * np.cos(DH[5, 1]), np.sin(DH[5, 0]) * np.sin(DH[5, 1]),
                    DH[5, 2] * np.cos(DH[5, 0])],
                   [np.sin(DH[5, 0]), np.cos(DH[5, 0]) * np.cos(DH[5, 1]), -np.cos(DH[5, 0]) * np.sin(DH[5, 1]),
                    DH[5, 2] * np.sin(DH[5, 0])],
                   [0, np.sin(DH[5, 1]), np.cos(DH[5, 1]), DH[5, 3]],
                   [0, 0, 0, 1]])

    H0_6 = np.dot(H0_5, H5_6)

    R0_0M = np.matrix([H0_0[0, 2], H0_0[1, 2], H0_0[2, 2]])
    R0_1M = np.matrix([H0_1[0, 2], H0_1[1, 2], H0_1[2, 2]])
    R0_2M = np.matrix([H0_2[0, 2], H0_2[1, 2], H0_2[2, 2]])
    R0_3M = np.matrix([H0_3[0, 2], H0_3[1, 2], H0_3[2, 2]])
    R0_4M = np.matrix([H0_4[0, 2], H0_4[1, 2], H0_4[2, 2]])
    R0_5M = np.matrix([H0_5[0, 2], H0_5[1, 2], H0_5[2, 2]])

    D0_6__D0_0 = np.matrix([H0_6[0, 3], H0_6[1, 3], H0_6[2, 3]]) - np.matrix([H0_0[0, 3], H0_0[1, 3], H0_0[2, 3]])
    D0_6__D0_1 = np.matrix([H0_6[0, 3], H0_6[1, 3], H0_6[2, 3]]) - np.matrix([H0_1[0, 3], H0_1[1, 3], H0_1[2, 3]])
    D0_6__D0_2 = np.matrix([H0_6[0, 3], H0_6[1, 3], H0_6[2, 3]]) - np.matrix([H0_2[0, 3], H0_2[1, 3], H0_2[2, 3]])
    D0_6__D0_3 = np.matrix([H0_6[0, 3], H0_6[1, 3], H0_6[2, 3]]) - np.matrix([H0_3[0, 3], H0_3[1, 3], H0_3[2, 3]])
    D0_6__D0_4 = np.matrix([H0_6[0, 3], H0_6[1, 3], H0_6[2, 3]]) - np.matrix([H0_4[0, 3], H0_4[1, 3], H0_4[2, 3]])
    D0_6__D0_5 = np.matrix([H0_6[0, 3], H0_6[1, 3], H0_6[2, 3]]) - np.matrix([H0_5[0, 3], H0_5[1, 3], H0_5[2, 3]])

    # cross multiplying the matrices
    R0_0D = np.cross(R0_0M, D0_6__D0_0)
    R0_1D = np.cross(R0_1M, D0_6__D0_1)
    R0_2D = np.cross(R0_2M, D0_6__D0_2)
    R0_3D = np.cross(R0_3M, D0_6__D0_3)
    R0_4D = np.cross(R0_4M, D0_6__D0_4)
    R0_5D = np.cross(R0_5M, D0_6__D0_5)

    R0_0D = np.reshape(R0_0D, (3, 1))
    R0_1D = np.reshape(R0_1D, (3, 1))
    R0_2D = np.reshape(R0_2D, (3, 1))
    R0_3D = np.reshape(R0_3D, (3, 1))
    R0_4D = np.reshape(R0_4D, (3, 1))
    R0_5D = np.reshape(R0_5D, (3, 1))

    R0_0M = np.reshape(R0_0M, (3, 1))
    R0_1M = np.reshape(R0_1M, (3, 1))
    R0_2M = np.reshape(R0_2M, (3, 1))
    R0_3M = np.reshape(R0_3M, (3, 1))
    R0_4M = np.reshape(R0_4M, (3, 1))
    R0_5M = np.reshape(R0_5M, (3, 1))

    # J is the jacobian matrix
    J_upper = np.concatenate((R0_0D, R0_1D, R0_2D, R0_3D, R0_4D, R0_5D), axis= 1)
    J_lower = np.concatenate((R0_0M, R0_1M, R0_2M, R0_3M, R0_4M, R0_5M), axis= 1)

    J = np.concatenate((J_upper, J_lower))
    # print(J)
    Jinv = np.linalg.inv(np.matrix(J, dtype='float'))

    vend = np.matrix([[vx], [vy], [vz], [valpha], [vbeta], [vgama]])

    # speed of rotation of joints
    Vq =  np.dot(Jinv, vend)

    return Vq[0, 0], Vq[1, 0], Vq[2, 0], Vq[3, 0], Vq[4, 0], Vq[5, 0]


# it will move the end effector at ginven speed for given time in ms
def go_with_speed(px, py, pz, roll, pitch, yaw, vx , vy, vz, valpha, vbeta, vgama, time):

    fig, ax = create_plot()

    # getting initial angles
    q1, q2, q3, q4, q5, q6 = get_angles(px, py, pz, roll, pitch, yaw)

    X = []; Y = []; Z = []; X1 = []; Y1 = []; Z1 = []

    # step counter, each step of 1 ms, if want to change, pls change the value of plt.pause() in update_plot function also
    i=0

    



    while i<=time:
        # getting speed at each iteration
        vq1, vq2, vq3, vq4, vq5, vq6 = get_speed(vx , vy, vz, valpha, vbeta, vgama, q1, q2, q3, q4, q5, q6)

        # calculating new values on the basis of speed, after 1 ms, change the value of dt to change the interval
        q1 = q1 + vq1*dt; q2 = q2 + vq2*dt; q3 = q3 + vq3*dt; q4 = q4 + vq4*dt; q5 = q5 + vq5*dt; q6 = q6 + vq6*dt

        # rest is for plotting
        X, Y, Z = forward_kin(q1, q2, q3, q4, q5, q6)
        X1.append(X[0, 7]); Y1.append(Y[0, 7]); Z1.append(Z[0, 7])
        X11 = X1[:]; Y11 = Y1[:]; Z11 = Z1[:]
        X11 = np.reshape(X1, (1, i + 1))
        Y11 = np.reshape(Y1, (1, i + 1))
        Z11 = np.reshape(Z1, (1, i + 1))
        update_plot(X, Y, Z, X11, Y11, Z11, fig, ax)
        i+=1

    print(X[0, 7], Y[0, 7], Z[0, 7])
    plt.show()


# we have to define the parametric form of the trajectory we want to follow
def get_trajectory(y1,y2,z1,z2,t):
    # define the trajectory
   
    
    m=np.sqrt( (y2-y1)*(y2-y1) +(z2-z1)*(z2-z1))

    v= 23.0 # we can control the speed by this variable

    px = 4.0
    py = ((v*(y2-y1))/m )*t +y1
    pz = ((v*(z2-z1))/m )*t +z1


    # just put the derivative of the trajectory to gain the speed
    vx = 0
    vy = ((v*(y2-y1))/m )
    vz = ((v*(z2-z1))/m )

    return px, py, pz, vx, vy, vz


def follow_trajectory():
    fig, ax = create_plot()
    px=4.0
    py=10.0
    pz=1.0

    

    # getting initial point of tregectory
    #px, py, pz, vx, vy, vz = get_trajectory(py,pz,0)


    # declaring the pose and velocity of the end effector
    roll=0; pitch=0; yaw=0; valpha=0; vbeta=0; vgama=0

    q1, q2, q3, q4, q5, q6 = get_angles(px, py, pz, roll, pitch, yaw)



    X = []; Y = []; Z = []; X1 = []; Y1 = []; Z1 = []
    i = 0
    t = 0  # parametric variable

    Y0 = [10,8.5,7,7.75,9.25,7.75,7,6,6,4.5,4.5,6,4.5,3.5,2,2.75,3.5,2.75,3.5,2.5,1,-0.5,0.25,1.75,0.25,-0.5]
    Z0 = [1,4,1,2.5,2.5,2.5,1,1,4,4,3,3,1,1,4,2.5,4,2.5,1,1,4,1,2.5,2.5,2.5,1]

    n=0

    for j in range(25):

        py, pz = Y0[j], Z0[j]
        h , k = Y0[j+1], Z0[j+1]

        #i=0
        t=0

        while abs (pz-k) > 1.4 or abs(py-h)>0.9:

            # getting the value of speed inf position after each iteration according to specified trajectory
            px, py, pz, vx, vy, vz = get_trajectory(py,h,pz,k,t)

            q1, q2, q3, q4, q5, q6 = get_angles(px, py, pz, roll, pitch, yaw)

            print (px)
            print (py)
            print (pz)
            print(" ")

            # getting values of each joints according to points
        

            # speed of the angles
            vq1, vq2, vq3, vq4, vq5, vq6 = get_speed(vx, vy, vz, valpha, vbeta, vgama, q1, q2, q3, q4, q5, q6)

             # calculating new values on the basis of speed, after 1 ms, change the value of dt to change the interval
            q1 = q1 + vq1 * dt; q2 = q2 + vq2 * dt; q3 = q3 + vq3 * dt; q4 = q4 + vq4 * dt; q5 = q5 + vq5 * dt; q6 = q6 + vq6 * dt

            # rest is for plotting
            X, Y, Z = forward_kin(q1, q2, q3, q4, q5, q6)
            X1.append(X[0, 7]); Y1.append(Y[0, 7]); Z1.append(Z[0, 7])
            X11 = X1[:]; Y11 = Y1[:]; Z11 = Z1[:]
            X11 = np.reshape(X1, (1, i + 1))
            Y11 = np.reshape(Y1, (1, i + 1))
            Z11 = np.reshape(Z1, (1, i + 1))
            update_plot(X, Y, Z, X11, Y11, Z11, fig, ax)
            i += 1

            n+=1

            # we can further decrease in the incrementation of parametric variable to make the trajectory smoother
            t += .03

            #print(X[0, 7], Y[0, 7], Z[0, 7])

        print("hello")
        print("clear")
    print(n)


def main():
    # define the position of the end efector
    px, py, pz = 5.0, 1.0, 1.0
    # value of orientation of the end effector
    roll, pitch, yaw = 0, 0, 0
    # q1, q2, q3, q4, q5, q6 = get_angles(px, py, pz, roll, pitch, yaw)

    # to print a specific word, curently specified as "MNNIT" in that function itself
    print_word()

    # to move the end effector in a particular direction with specific speed for a specified time
    #vx = 0; vy = -100; vz = 0; valpha = 0; vbeta = 0; vgama = 0; time = 10  # time in ms
    #go_with_speed(px, py, pz, roll, pitch, yaw, vx , vy, vz, valpha, vbeta, vgama, time)

    # to follow a trajectory specified in get_trajectory function
    #follow_trajectory()


if __name__ == "__main__":
    main()