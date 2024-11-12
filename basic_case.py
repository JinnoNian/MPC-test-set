#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:43:21 2024

@author: constantinesun
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# System parameters
dt = 0.1  # Time step
N = 50  # Prediction horizon
x0 = np.array([1, 1, 0, 0])  # Initial state
XD = np.array([5, 5, 0, 0])  # Desired final state
v_max = 5  # Velocity limit
a_max = 5  # Acceleration limit

# State space model matrices
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[0], [0], [dt], [dt]])

# Weight matrices for the cost function
Q = np.diag([10, 10, 1, 1])  # State error penalty
R = np.array([[1]])  # Control effort penalty

# Optimization variables
x = cp.Variable((4, N + 1))
u = cp.Variable((1, N))

# Cost function and constraints
cost = 0
constraints = [x[:, 0] == x0]
for k in range(N):
    cost += cp.quad_form(x[:, k] - XD, Q) + cp.quad_form(u[:, k], R)
    constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
    constraints += [cp.norm(u[:, k], 'inf') <= a_max]
    constraints += [cp.norm(x[2:4, k], 'inf') <= v_max]  # Velocity constraints

# Terminal cost
cost += cp.quad_form(x[:, N] - XD, Q)

# Solve the optimization problem
prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve()

# Extract the optimal trajectory
x_opt = x.value

# Plot the trajectory
plt.figure(figsize=(10, 6))
plt.plot(x_opt[0, :], x_opt[1, :], 'b-o', label='Optimal Trajectory')
plt.plot(XD[0], XD[1], 'rx', markersize=10, label='Target Position')
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('MPC Optimal Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Plot position over time
plt.figure(figsize=(10, 6))
plt.plot(np.arange(N + 1) * dt, x_opt[0, :], 'b-', label='x position')
plt.plot(np.arange(N + 1) * dt, x_opt[1, :], 'r-', label='y position')
plt.xlabel('Time (s)')
plt.ylabel('Position')
plt.title('Position Over Time')
plt.legend()
plt.grid(True)
plt.show()
