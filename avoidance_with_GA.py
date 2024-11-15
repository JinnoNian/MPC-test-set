#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:46:58 2024

@author: constantinesun
"""
# -*- coding: utf-8 -*-
"""
MPC with Genetic Algorithm Optimization for Obstacle Avoidance
"""

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import casadi as ca

# Parameters
dt = 0.1  # time step
N = 20  # prediction horizon
max_velocity = 5
max_acceleration = 5
xd = np.array([5, 5])  # target destination

# Weights for cost function
Q = np.diag([10, 10, 1, 1])  # state penalty
R = np.diag([1, 1])  # control penalty

# Initial conditions
x0 = np.array([1, 1, 0, 0])

# State-space model
A = np.array(
    [[1, 0, dt, 0],
     [0, 1, 0, dt],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])
B = np.array([[0, 0],
              [0, 0],
              [dt, 0],
              [0, dt]])

# Define obstacle boundaries (circle)
obstacle_center = np.array([3, 3])
obstacle_radius = 1  # Radius of the circular obstacle

# Cost function
def evaluate(U_flat, X_init):
    U = U_flat.reshape((N, 2))
    X = X_init.copy()
    cost = 0
    
    for k in range(N):
        U_k = U[k]
        X = A @ X + B @ U_k
        
        # Calculate cost
        cost += (xd - X[:2]).T @ Q[:2, :2] @ (xd - X[:2]) + U_k.T @ R @ U_k
        
        # Obstacle avoidance constraint (penalty if within obstacle)
        if (X[0] - obstacle_center[0])**2 + (X[1] - obstacle_center[1])**2 < obstacle_radius**2:
            cost += 1000  # Large penalty for being inside the obstacle
    
    # Terminal cost
    cost += np.linalg.norm(X[:2] - xd)
    
    return cost,

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -max_acceleration, max_acceleration)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, N * 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", lambda ind: evaluate(ind, X_current))

# Genetic Algorithm parameters
population_size = 100
generations = 50
crossover_prob = 0.7
mutation_prob = 0.2

# Rolling Horizon Optimization
X_current = x0.copy()
trajectory = [X_current[:2]]
iteration = 0

while iteration < 100 and np.linalg.norm(X_current[:2] - xd) > 0.1:
    # Run Genetic Algorithm
    population = toolbox.population(n=population_size)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    
    result_population, _ = algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                                              ngen=generations, verbose=False)
    
    # Select the best individual
    best_individual = tools.selBest(result_population, k=1)[0]
    optimal_u = best_individual.reshape((N, 2))
    
    # Apply the first control input
    U_k = optimal_u[0]
    X_current = A @ X_current + B @ U_k
    trajectory.append(X_current[:2])
    
    iteration += 1

# Convert trajectory to numpy array for plotting
tra = np.array([np.array(point).flatten() for point in trajectory])

print(tra)

# Plot trajectory
plt.plot(tra[:, 0], tra[:, 1], '-o', label='Trajectory')
plt.scatter(x0[0], x0[1], color='green', label='Start', marker='x')
plt.scatter(xd[0], xd[1], color='red', label='Target', marker='*')
plt.scatter(obstacle_center[0], obstacle_center[1], color='blue', label='Obstacle Center', marker='s')
circle = plt.Circle(obstacle_center, obstacle_radius, color='blue', alpha=0.3, label='Obstacle Area')
plt.gca().add_patch(circle)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('MPC Trajectory with Obstacle Avoidance (GA Optimization)')
plt.grid(True)
plt.axis('equal')
plt.show()
