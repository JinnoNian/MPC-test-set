#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:46:58 2024

@author: constantinesun
"""
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.1  # time step
N = 30  # prediction horizon
max_velocity = 5
max_acceleration = 5
xd = np.array([5, 5])  # target destination
obt = np.array([3, 3])  # obstacle position
epsilon = 0.5  # safety distance

# Weights for cost function
Q = np.diag([10, 10, 1, 1])  # state penalty
R = 1  # control penalty

# Initial conditions
x0 = np.array([1, 1, 0, 0])

# Define optimization variables
x = ca.SX.sym('x', 4)
u = ca.SX.sym('u')

# State-space model
A = ca.DM(
    [[1, 0, dt, 0],
     [0, 1, 0, dt],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])
B = ca.DM([0, 0, dt, dt])

# Objective function and constraints
U = ca.SX.sym('U', N)
X = x0
cost = 0
constraints = []
trajectory = [x0[:2]]

for k in range(N):
    # Update state
    X = A @ X + B * U[k]
    trajectory.append(X[:2])
    
    # Calculate cost
    cost += ca.mtimes((xd - X[:2]).T, Q[:2, :2] @ (xd - X[:2])) + R * U[k]**2
    
    # Constraints
    constraints.append(U[k] <= max_acceleration)
    constraints.append(U[k] >= -max_acceleration)
    constraints.append(ca.mtimes((X[:2] - obt).T, (X[:2] - obt)) >= epsilon**2)

# Create optimization problem
opts = {'ipopt.print_level': 0, 'print_time': 0}
nlp = {'x': U, 'f': cost, 'g': ca.vertcat(*constraints)}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Solve the MPC problem
solution = solver(lbx=-max_acceleration, ubx=max_acceleration, lbg=0, ubg=np.inf)
optimal_u = solution['x'].full().flatten()

# Simulate trajectory
X = x0
trajectory = [X[:2]]
for k in range(N):
    X = A @ X + B * optimal_u[k]
    trajectory.append(X[:2])

tra = [np.array(point).flatten() for point in trajectory]
tra = np.array(tra)

print(tra)

# Plot trajectory
plt.plot(tra[:, 0], tra[:, 1], '-o', label='Trajectory')
plt.scatter(x0[0], x0[1], color='green', label='Start', marker='x')
plt.scatter(xd[0], xd[1], color='red', label='Target', marker='*')
plt.scatter(obt[0], obt[1], color='blue', label='Obstacle', marker='s')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('MPC Trajectory with Obstacle Avoidance')
plt.grid(True)
plt.axis('equal')
plt.show()
