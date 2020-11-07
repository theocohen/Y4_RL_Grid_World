#%%

import itertools
import matplotlib.pyplot as plt
import numpy as np

#%%

# Question 1.c.3 State Value
P = np.array([[0, 1, 0, 0],
              [0, 1 / 3, 1 / 3, 1 / 3],
              [0, 0, 1, 0],
              [1 / 2, 1 / 2, 0, 0]])

R = np.array([[0, 2, 0, 0],
              [0, 3, 3, 3],
              [0, 0, 0, 0],
              [1, 1, 0, 0]])

# np.linalg.inv(np.identity(4) - P)

def state_value(R, P, discount, threshold=0.001):
    # Compute v(s) for each state using Iterative Policy Evaluation

    epochs = 0

    # Make sure delta is bigger than the threshold to start with
    delta = 2 * threshold

    # The value is initialised at 0
    V = np.zeros(P.shape[0])

    # While the Value has not yet converged do:
    while delta > threshold:
        delta = 0
        epochs += 1
        for state_idx in range(P.shape[0]):

            v_old = V[state_idx]
            # Accumulator variable for the State-Action Value
            tmpQ = 0
            for state_idx_prime in range(P.shape[0]):
                tmpQ += P[state_idx, state_idx_prime] * (R[state_idx, state_idx_prime] + discount * V[state_idx_prime])

            # Update the value of the state
            V[state_idx] = tmpQ

        # After updating the values of all states, update the delta
        delta = max(delta, abs(v_old - V[state_idx]))

    return V, epochs

print("State value for each state: {}".format(state_value(R, P, 1)))

#%%
# Question 2 GridWorld

class GridWorld(object):

    def __init__(self):

        # Parameters of the GridWorld
        self.shape = (6, 6)
        self.obstacle_locs = [(1, 1), (2, 3), (2, 5), (3, 1), (4, 1), (4, 2), (4, 4)]
        self.absorbing_locs = [(1, 3), (4, 3)]
        self.special_rewards = [10, -100]  # Corresponds to each of the absorbing_locs in order
        self.default_reward = -1
        self.action_names = ['N', 'E', 'S', 'W']  # Action 0 is 'N', 1 is 'E' and so on
        self.actions_displacement = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # displacement corresponding to each action
        self.action_size = len(self.action_names)

        # probability of successfully going to where the action was aiming at (corresponds to 'p')
        self.action_success_proba = 9 / 20

        # Build attributes defining the GridWorld
        self._build_grid_world()

        # TODO change
        self.starting_loc = (3, 0)
        self.starting_state = self._loc_to_state(self.starting_loc, self.locs)  # Number of the starting state
        # Locating the initial state
        self.initial = np.zeros((1, len(self.locs)));
        self.initial[0, self.starting_state] = 1

        # Placing the walls on a bitmap
        self.walls = np.zeros(self.shape);
        for ob in self.obstacle_locs:
            self.walls[ob] = 1

        # Placing the absorbers on a grid for illustration
        self.absorbers = np.zeros(self.shape)
        for ab in self.absorbing_locs:
            self.absorbers[ab] = -1

        # Placing the rewarders on a grid for illustration
        self.rewarders = np.zeros(self.shape)
        for i, rew in enumerate(self.absorbing_locs):
            self.rewarders[rew] = self.special_rewards[i]

        # Illustrating the grid world
        self._paint_maps()

    ### Private methods

    def _paint_maps(self):
        """Helper function to print the grid word used"""
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(self.walls)
        plt.title('Obstacles')
        plt.subplot(1, 3, 2)
        plt.imshow(self.absorbers)
        plt.title('Absorbing states')
        plt.subplot(1, 3, 3)
        plt.imshow(self.rewarders)
        plt.title('Reward states')
        plt.show()

    def _build_grid_world(self):
        """Build GridWorld's internal attributes (refactored)"""

        # List of all valid states coordinates
        self.locs = [loc for loc in itertools.product(range(self.shape[0]), range(self.shape[1])) if self._is_valid_location(loc)]
        self.state_size = len(self.locs)  # Number of valid states in the GridWorld

        # Translate absorbing locations into absorbing state indices
        self.absorbing = [loc in self.absorbing_locs for loc in self.locs]

        # Transition matrix encoding the probability of T[st+1, st, a]
        self.T = np.zeros((self.state_size, self.state_size, self.action_size))

        # Reward matrix encoding R[st+1, st, a]
        self.R = self.default_reward * np.ones((self.state_size, self.state_size, self.action_size))

        # Fill the transition matrix
        for (prior_state, action) in itertools.product(range(self.state_size), range(self.action_size)):
            for neighbour_orientation, post_state in enumerate(self._get_neighbours(self.locs[prior_state])):
                if action == neighbour_orientation:
                    # if the post state is the one that was aimed by the action, then action is successful with probability p
                    proba = self.action_success_proba
                else:
                    # otherwise, the action failed and resulted in one of the neighbouring state
                    proba = (1 - self.action_success_proba) / (self.action_size - 1)
                self.T[post_state, prior_state, action] += proba  # summing each neighbour can be the prior_state

        # Fill the reward matrix
        for i, sr in enumerate(self.special_rewards):
            post_state = self._loc_to_state(self.absorbing_locs[i], self.locs)
            self.R[post_state, :, :] = sr

    def _loc_to_state(self, loc, locs):
        """Gives state index corresponding to a set of coordinates"""
        return locs.index(tuple(loc))

    def _is_valid_location(self, loc):
        """It is a valid location if it is in grid and not obstacle (refactored)"""
        return 0 <= loc[0] < self.shape[0] and 0 <= loc[1] < self.shape[1] and loc not in self.obstacle_locs

    def _get_neighbours(self, loc):
        """Returns list of neighbours (index) given a location (refactored)"""
        neighbours_coords = [(loc[0] + i, loc[1] + j) if self._is_valid_location((loc[0] + i, loc[1] + j)) else loc for (i, j) in self.actions_displacement]
        return list(map(lambda coords: self._loc_to_state(coords, self.locs), neighbours_coords))

    def _get_action_target_state(self, state, action):
        """Returns target state of an action

        :param state: prior state index
        :param action: action index performed
        :return: target state index
        """
        return self._loc_to_state(tuple(x + i for x, i in zip(self.locs[state], self.actions_displacement[action])), self.locs)

    ### Drawing Functions

    def draw_deterministic_policy(self, policy):
        """
        Draws a deterministic policy.
        The policy needs to be a np array of 22 values between 0 and 3 with.
        0 -> N, 1->E, 2->S, 3->W
        """
        plt.figure()

        plt.imshow(self.walls + self.rewarders + self.absorbers)  # Create the graph of the grid
        # plt.hold('on')
        for state, action in enumerate(policy):
            if self.absorbing[state]:  # If it is an absorbing state, don't plot any action
                continue
            arrows = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$",
                      r"$\leftarrow$"]  # List of arrows corresponding to each possible action
            action_arrow = arrows[action]  # Take the corresponding action
            location = self.locs[state]  # Compute its location on graph
            plt.text(location[1], location[0], action_arrow, ha='center', va='center')  # Place it on graph

        plt.show()

    def draw_value(self, Value):
        """
        Draws a policy value function.
        The value need to be a np array of 22 values
        """
        plt.figure()

        plt.imshow(self.walls + self.rewarders + self.absorbers)  # Create the graph of the grid
        for state, value in enumerate(Value):
            if (self.absorbing[state]):  # If it is an absorbing state, don't plot any value
                continue
            location = self.locs[state]  # Compute the value location on graph
            plt.text(location[1], location[0], round(value, 2), ha='center', va='center')  # Place it on graph

        plt.show()

    def draw_deterministic_policy_grid(self, policy, title, n_columns, n_lines):
        """Draws a grid of deterministic policy.
        The policy needs to be an array of np array of 22 values between 0 and 3 with
        0 -> N, 1->E, 2->S, 3->W."""
        plt.figure(figsize=(20, 8))
        for subplot in range(len(policy)):  # Go through all policies
            ax = plt.subplot(n_columns, n_lines, subplot + 1)  # Create a subplot for each policy
            ax.imshow(self.walls + self.rewarders + self.absorbers)  # Create the graph of the grid
            for state, action in enumerate(policy[subplot]):
                if (self.absorbing[state]):  # If it is an absorbing state, don't plot any action
                    continue
                arrows = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$",
                          r"$\leftarrow$"]  # List of arrows corresponding to each possible action
                action_arrow = arrows[action]  # Take the corresponding action
                location = self.locs[state]  # Compute its location on graph
                plt.text(location[1], location[0], action_arrow, ha='center', va='center')  # Place it on graph
            ax.title.set_text(title[subplot])  # Set the title for the graoh given as argument
        plt.show()

    def draw_value_grid(self, Value, title, n_columns, n_lines):
        """Draw a grid of value function.
        The value need to be an array of np array of 22 values"""
        plt.figure(figsize=(20, 8))
        for subplot in range(len(Value)):  # Go through all values
            ax = plt.subplot(n_columns, n_lines, subplot + 1)  # Create a subplot for each value
            ax.imshow(self.walls + self.rewarders + self.absorbers)  # Create the graph of the grid
            for state, value in enumerate(Value[subplot]):
                if (self.absorbing[state]):  # If it is an absorbing state, don't plot any value
                    continue
                location = self.locs[state]  # Compute the value location on graph
                plt.text(location[1], location[0], round(value, 1), ha='center', va='center')  # Place it on graph
            ax.title.set_text(title[subplot])  # Set the title for the graoh given as argument
        plt.show()

    ### Methods

    def policy_evaluation(self, policy, discount, threshold=0.0001):

        # Make sure delta is bigger than the threshold to start with
        delta = 2 * threshold

        # The value is initialised at 0
        V = np.zeros(policy.shape[0])
        # Make a deep copy of the value array to hold the update during the evaluation
        Vnew = np.copy(V)

        epoch = 0
        # While the Value has not yet converged do:
        while delta > threshold:
            epoch += 1
            for state_idx in range(policy.shape[0]):
                # If it is one of the absorbing states, ignore
                if (self.absorbing[state_idx]):
                    continue

                # Accumulator variable for the Value of a state
                tmpV = 0
                for action_idx in range(policy.shape[1]):
                    # Accumulator variable for the State-Action Value
                    tmpQ = 0
                    for state_idx_prime in range(policy.shape[0]):
                        tmpQ += self.T[state_idx_prime, state_idx, action_idx] * (
                                    self.R[state_idx_prime, state_idx, action_idx] + discount * V[state_idx_prime])
                    tmpV += policy[state_idx, action_idx] * tmpQ

                # Update the value of the state
                Vnew[state_idx] = tmpV

            # After updating the values of all states, update the delta
            delta = max(abs(Vnew - V))
            # and save the new value into the old
            V = np.copy(Vnew)

        return V, epoch

    def policy_iteration(self, discount, threshold=0.0001):
        # 1. Init
        policy = np.eye(grid.action_size)[[0] * grid.state_size]  # one-hot encoding of shape (state_size, action_size)
        policy_stable = False

        epochs = 0
        while not policy_stable:

            # 2. Policy Evaluation
            V, eval_epochs = grid.policy_evaluation(policy, threshold, discount)
            epochs += eval_epochs  # Count epochs from evaluation too
            # print("Epoch nbr {}".format(epochs))

            # 3. Policy Improvement
            policy_stable = True
            for state_idx in range(grid.state_size):

                # If it is one of the absorbing states, ignore
                if not grid.absorbing[state_idx]:

                    # Which action does the current policy choose from a given state?
                    prev_action_idx = np.argmax(policy[state_idx])

                    # What is the best action from given state? Q value
                    Q = np.zeros(grid.action_size)
                    for state_idx_prime in range(grid.state_size):
                        Q += self.T[state_idx_prime, state_idx, :] * (
                                    self.R[state_idx_prime, state_idx, :] + discount * V[state_idx_prime])

                    policy[state_idx] = np.eye(grid.action_size)[np.argmax(Q)]

                    # Does it improve current policy?
                    policy_stable = policy_stable and (prev_action_idx == np.argmax(policy[state_idx]))

        return policy, V, epochs

    def value_iteration(self, discount, threshold=0.0001):
        # 1. Init
        V = np.zeros(grid.state_size)

        epochs = 0
        delta = threshold
        while delta >= threshold:
            delta = 0
            epochs += 1

            for state_idx in range(grid.state_size):

                # If it is one of the absorbing states, ignore
                if not grid.absorbing[state_idx]:

                    # current state value
                    v = V[state_idx]

                    # What is the best action from given state? Q value
                    Q = np.zeros(grid.action_size)
                    for state_idx_prime in range(grid.state_size):
                        Q += self.T[state_idx_prime, state_idx, :] * (
                                    self.R[state_idx_prime, state_idx, :] + discount * V[state_idx_prime])
                    V[state_idx] = max(Q)

                    # New delta
                    delta = max(delta, abs(v - V[state_idx]))

        # Build policy
        optimal_policy = np.zeros((grid.state_size, grid.action_size))  # Initialisation

        for state_idx in range(grid.state_size):

            # Compute Q value
            Q = np.zeros(grid.action_size)
            for state_idx_prime in range(grid.state_size):
                Q += self.T[state_idx_prime, state_idx, :] * (
                            R[state_idx_prime, state_idx, :] + discount * V[state_idx_prime])

            optimal_policy[state_idx, np.argmax(Q)] = 1

        return optimal_policy, grid.policy_evaluation(optimal_policy, threshold, discount)[0], epochs


grid = GridWorld()
print(grid.T)
print(grid.R)
## Question 2.b Dynamic Programming (Value Iteration?)
