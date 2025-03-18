#! /usr/bin/env python

"""
Used to compute Dynamic Programming (DP) policy evaluation and policy iteration 
within a grid world where the only movements possible are up, down and right.

"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import array, zeros
from numpy.ma import masked_array
from random import randint

class PolicyGrid:

    def __init__(self, part_a_scenario=True, width=3, height=4, obstacle_prob=0):

        if part_a_scenario:
            # Initialize grid world with the scenario from Part A.
            self.size = (3, 4)
            self.occupancy = zeros(self.size)
            self.occupancy[0,2] = 1
            self.occupancy[1,1] = 1
            self.occupancy[1,2] = 1

            # Set policy 0 as in Part A
            self.policy_y_array = array([[1, 1, 0, -1],
                                         [1, 0, 0, -1],
                                         [1, 1, -1, -1]])
            self.policy_x_array = array([[0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0]])

            # Set reward function.  As a slight variation from the notes, we'll
            # assign rewards directly to states.
            self.rewards = array([[0, 0, 0, 10],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])
        else:
            self.size = (height, width)
            # Initialize with each cell having obstacle_prob probability
            # of being occupied.  0 = free.  1 = occupied
            self.occupancy = np.random.binomial(1, obstacle_prob, width*height).reshape(self.size)
            
            self.set_random_policy()

            # See comment above about reward function.  Here we just pick the
            # upper-right corner as the only rewarded state.
            self.rewards = zeros(self.size)
            self.rewards[0,-1] = 10

        # Discount factor for policy evaluation and iteration.
        self.gamma = 0.9
        self.values = zeros(self.size)

        print("self.occupancy:")
        print(self.occupancy)
        print("self.policy_y_array:")
        print(self.policy_y_array)
        print("self.policy_x_array:")
        print(self.policy_x_array)
        print("self.rewards:")
        print(self.rewards)

    def future_state(self, y, x, dy, dx):
        """Return the future state given the current state and action."""
        new_y = y + dy
        new_x = x + dx
        if new_y < 0 or new_y >= self.size[0] or new_x < 0 or new_x >= self.size[1]:
            new_y = y
            new_x = x
        if self.occupancy[new_y, new_x] == 1:
            new_y = y
            new_x = x
        return new_y, new_x
    def set_random_policy(self):
        self.policy_y_array = zeros(self.size, dtype=np.int32)
        self.policy_x_array = zeros(self.size, dtype=np.int32)
        for j in range(self.size[0]):
            for i in range(self.size[1]):
                if self.occupancy[j,i] == 1:
                    continue

                r = randint(0, 2)
                if r == 0:
                    # Up
                    self.policy_y_array[j,i] = -1
                    self.policy_x_array[j,i] = 0
                elif r == 1:
                    # Down
                    self.policy_y_array[j,i] = 1
                    self.policy_x_array[j,i] = 0
                else:
                    # Right
                    self.policy_y_array[j,i] = 0
                    self.policy_x_array[j,i] = 1

    def draw_policy(self, ax):
        # First mask out zero actions.
        mask = zeros(self.size)
        for j in range(self.size[0]):
            for i in range(self.size[1]):
                dy = self.policy_y_array[j,i]
                dx = self.policy_x_array[j,i]
                if dy == 0 and dx == 0:
                    mask[j,i] = 1

        masked_y_array = masked_array(self.policy_y_array, mask)
        masked_x_array = masked_array(self.policy_x_array, mask)
        ax.quiver(masked_x_array, -masked_y_array)

    def draw(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)

        ax1.set_title("Occupancy", y=-10.1)
        ax1.matshow(self.occupancy)

        ax2.set_title("Reward")
        ax2.matshow(self.rewards)

        ax3.set_title("Value Function + Policy")
        value_image = ax3.matshow(self.values)
        #fig.colorbar(value_image, ax=ax3)
        self.draw_policy(ax3)

        plt.show()

    def evaluate_policy(self, max_iterations=100, convergence_threshold=0.1):
        for i in range(max_iterations):
            old_values = self.values.copy()
            for j in range(len(self.occupancy)):
                for k in range(len(self.occupancy[j])):
                    if(self.occupancy[j][k]==0):
                        self.values[j][k] = self.rewards[j][k]+self.gamma *old_values[self.future_state(j, k, self.policy_y_array[j][k], self.policy_x_array[j][k])[0]][self.future_state(j, k, self.policy_y_array[j][k], self.policy_x_array[j][k])[1]]
                print(self.values[j])
        ################################################################
        # an implementation
        # for DP policy evaluation.  The idea is to evaluate the current policy
        # which exists in self.policy_y_array and self.policy_x_array.
        
    def Q_value(self, y, x, a):
        return self.rewards[y][x] + self.gamma * self.values[self.future_state(y, x, a[0], a[1])[0]][self.future_state(y, x, a[0], a[1])[1]]
    def policy_improvement(self):
        """Complete one step of policy improvement based on the existing value
        function in self.values (must be computed prior to this)."""
        new_policy = [None for _ in range(5)]
        for y in range(len(self.occupancy)):
            for x in range(len(self.occupancy[y])):
                if self.occupancy[y][x] == 1:
                    continue
                Q_values = [self.Q_value(y, x, a) for a in [(-1, 0), (1, 0), (0, 1)]]
                self.policy_y_array[y][x] = [(-1, 0), (1, 0), (0, 1)][np.argmax(Q_values)][0]
                self.policy_x_array[y][x] = [(-1, 0), (1, 0), (0, 1)][np.argmax(Q_values)][1]
        
        ################################################################
        # an implementation
        # for DP policy improvement.
        pass

    def policy_iteration(self):
        done = False
        while(not done):
            self.evaluate_policy()
            old_policy_y_array, old_policy_x_array = self.policy_y_array.copy(), self.policy_x_array.copy()
            self.policy_improvement()
            if (old_policy_y_array == self.policy_y_array).all() and (old_policy_x_array == self.policy_x_array).all():
                done = True
        ################################################################
        # an implementation
        # for DP policy improvement.
        pass



# STUDENTS: To test policy evaluation alone.
grid = PolicyGrid()
grid.evaluate_policy(max_iterations=3)

# STUDENTS: Uncomment this when you are ready to test policy improvement.
# You will have to print out the values and policy to see the results.
grid.policy_improvement()
print("Policy Improvement", grid.policy_y_array, grid.policy_x_array)
# STUDENTS: Uncomment this when you are ready to test full policy iteration
grid.policy_iteration()
grid.draw()

# STUDENTS: Uncomment the following for a larger test environment.
grid = PolicyGrid(part_a_scenario=False, width=20, height=15, obstacle_prob=0.1)
grid.policy_iteration()
grid.draw()
