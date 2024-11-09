#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:57:39 2024

@author: constantinesun
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# Parameter settings
N = 30  # Number of time steps
dt = 0.05  # Time step size
xd = np.array([2, 1])  # Target position
x_init = np.array([0, 1])  # Initial position of the manipulator
xdot_init = np.array([0.1, 0.1])  # Initial velocity
p_init = np.array([0, 0])  # Initial position of the obstacle
p_velocity = np.array([0.25, 0.25])  # Obstacle velocity
Q = np.eye(2)  # State weight matrix
R = 0.1  # Control weight
v_max = 1  # Maximum velocity

epsilon = 0.1  # Safety distance

# Initialization of state and control variables
x = np.zeros((N + 1, 2))
x[0] = x_init
xdot = np.zeros((N + 1, 2))
xdot[0] = xdot_init
u = np.zeros((N, 2))

# Initialization of obstacle position
p = np.zeros((N + 1, 2))
p[0] = p_init
for i in range(1, N + 1):
    if np.all(p[i - 1] < [1, 1]):
        p[i] = p[i - 1] + p_velocity
    else:
        p[i] = p[i - 1]

# Construct QP problem
P = np.zeros((2 * N, 2 * N))
q = np.zeros(2 * N)
A = []
b = []

# Objective function matrix P and vector q
for i in range(N):
    P[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R * np.eye(2)
    q[2 * i:2 * i + 2] = -2 * Q @ xd

# Add terminal state cost
P_terminal = 20 * Q  # Increase terminal state weight
P[-2:, -2:] += P_terminal

# Constraint matrices A and vector b
for i in range(N):
    # System dynamics constraints for velocity and position updates
    Ai_dyn = np.zeros((2, 2 * N))
    # Velocity update constraint: xdot[i+1] = xdot[i] + u[i] * dt
    Ai_dyn[0, 2 * i:2 * i + 2] = -dt
    Ai_dyn[1, 2 * i:2 * i + 2] = -dt
    A.append(Ai_dyn[0])
    A.append(Ai_dyn[1])
    b.append(0)
    b.append(0)
for i in range(N - 1):
    # System dynamics constraints for state evolution
    Ai_state = np.zeros((2, 2 * N))
    # Position update constraint: x[i+1] = x[i] + xdot[i] * dt
    Ai_state[0, 2 * i] = -1
    Ai_state[1, 2 * i + 1] = -1
    Ai_state[0, 2 * (i + 1)] = 1
    Ai_state[1, 2 * (i + 1) + 1] = 1
    A.append(Ai_state[0])
    A.append(Ai_state[1])
    b.append(0)
    b.append(0)

    # Velocity constraints
    Ai = np.zeros((1, 2 * N))
    Ai[0, 2 * i:2 * i + 2] = np.array([1, 1])
    A.append(Ai)
    b.append(v_max)
    # Obstacle distance constraint to ensure a safety distance from the obstacle
    Ai_dist = np.zeros((1, 2 * N))
    if i < N - 1:
        diff = x[i] - p[i + 1]
    else:
        diff = x[i] - p[i]  # Use current position for the last step to avoid out of bounds
    Ai_dist[0, 2 * i:2 * i + 2] = -2 * diff
    A.append(Ai_dist)
    b.append(-epsilon ** 2 + np.dot(diff, diff))

P = matrix(P)
q = matrix(q)
A = matrix(np.vstack(A))
b = matrix(np.array(b))

# Use QP solver
sol = solvers.qp(P, q, A, b)
if sol['status'] == 'optimal':
    u_opt = np.array(sol['x']).reshape(N, 2)
    # Update state trajectory
    for i in range(N):
        xdot[i + 1] = xdot[i] + u_opt[i] * dt
        x[i + 1] = x[i] + xdot[i] * dt
else:
    print("Optimization did not converge")

# Plot trajectory
plt.plot(x[:, 0], x[:, 1], 'b-o', label='Manipulator Trajectory')
plt.plot(p[:, 0], p[:, 1], 'r--', label='Obstacle Trajectory')
plt.plot(xd[0], xd[1], 'gx', markersize=10, label='Target Position')
plt.xlabel('X Direction')
plt.ylabel('Y Direction')
plt.legend()
plt.grid(True)
plt.title('Manipulator and Obstacle Trajectory')
plt.show()
