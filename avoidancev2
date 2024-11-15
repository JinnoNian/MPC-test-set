#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:54:04 2024

@author: constantinesun
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import cvxpy as cp

# Define start and goal points
start = np.array([1, 1])
goal = np.array([5, 5])
obstacle_center = np.array([3, 3])
obstacle_radius = 1.0

# RRT parameters
step_size = 0.5
max_iterations = 1000
goal_threshold = 0.5
x_limits = [0, 6]
y_limits = [0, 6]

# Define Node class
class Node:
    def __init__(self, position, parent=None):
        self.position = np.array(position)
        self.parent = parent

# Check if point collides with obstacle
def is_collision(point):
    return np.linalg.norm(point - obstacle_center) <= obstacle_radius

# Generate random node
def get_random_node():
    return np.array([random.uniform(x_limits[0], x_limits[1]), random.uniform(y_limits[0], y_limits[1])])

# Find the nearest node
def get_nearest_node(nodes, random_node):
    distances = [np.linalg.norm(node.position - random_node) for node in nodes]
    nearest_index = np.argmin(distances)
    return nodes[nearest_index]

# RRT algorithm implementation
def rrt(start, goal, max_iterations, step_size):
    start_node = Node(start)
    goal_node = Node(goal)
    nodes = [start_node]

    for _ in range(max_iterations):
        random_node = get_random_node()
        nearest_node = get_nearest_node(nodes, random_node)
        direction = random_node - nearest_node.position
        distance = np.linalg.norm(direction)
        direction = (direction / distance) * min(step_size, distance)
        new_position = nearest_node.position + direction

        if not is_collision(new_position):
            new_node = Node(new_position, nearest_node)
            nodes.append(new_node)

            # Check if the goal region is reached
            if np.linalg.norm(new_position - goal) <= goal_threshold:
                goal_node.parent = new_node
                nodes.append(goal_node)
                break

    # Backtrack the path
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.position)
        current_node = current_node.parent
    path.reverse()
    return path

# Run RRT algorithm and get path
path = rrt(start, goal, max_iterations, step_size)

# Divide the path into N reference points
N = 100  # Prediction horizon length
path_length = len(path)
reference_points = []
for i in range(N):
    index = int(i * (path_length - 1) / (N - 1))
    reference_points.append(path[index])

# State-space system parameters
dt = 0.1
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[0, 0],
              [0, 0],
              [dt, 0],
              [0, dt]])
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# MPC parameters
Q = np.eye(4)  # State error weight
R = np.eye(2)  # Control input weight
P = np.eye(4)  # Terminal cost weight

# Initial state
x = np.array([1, 1, 0, 0])
trajectory = [x[:2]]

# MPC to track the path
for ref in reference_points:
    # Define optimization variables
    u = cp.Variable((2, N))
    x_var = cp.Variable((4, N + 1))

    # Initial state constraint
    constraints = [x_var[:, 0] == x]

    # Dynamic constraints
    for k in range(N):
        constraints += [x_var[:, k + 1] == A @ x_var[:, k] + B @ u[:, k]]

    # Input constraints (velocity and acceleration limits to 5)
    constraints += [cp.abs(u) <= 5]
    constraints += [cp.abs(x_var[2:, :]) <= 5]  # Limit velocity and acceleration

    # Objective function
    cost = 0
    for k in range(N):
        cost += cp.quad_form(x_var[:, k] - np.append(ref, [0, 0]), Q) + cp.quad_form(u[:, k], R)
    # Terminal cost
    cost += cp.quad_form(x_var[:, N] - np.append(ref, [0, 0]), P)

    # Solve QP problem
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    # Get optimal control input and update state
    u_opt = u[:, 0].value
    x = A @ x + B @ u_opt
    trajectory.append(x[:2])

# Plot obstacle, start, and goal points
fig, ax = plt.subplots()
circle = plt.Circle(obstacle_center, obstacle_radius, color='r', alpha=0.5, label='Obstacle')
ax.add_patch(circle)
plt.plot(start[0], start[1], 'go', label='Start')
plt.plot(goal[0], goal[1], 'bo', label='Goal')

# Extract x and y coordinates from the path
path_x = [point[0] for point in path]
path_y = [point[1] for point in path]

# Plot RRT path
plt.plot(path_x, path_y, 'k--', label='RRT Planned Path')

# Plot MPC tracked trajectory
trajectory_x = [point[0] for point in trajectory]
trajectory_y = [point[1] for point in trajectory]
plt.plot(trajectory_x, trajectory_y, 'b-', label='MPC Tracked Path')

# Set plot attributes
plt.xlabel('X')
plt.ylabel('Y')
plt.title('RRT Path Planning with MPC Tracking (Velocity and Acceleration Limits)')
plt.grid()
plt.legend()
plt.axis('equal')
plt.show()
