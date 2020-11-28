import numpy as np
from numpy.linalg import pinv


class LinUCB(object):
    def __init__(self, name, d, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, color):
        """
        param:
            - d: dimension of the action vectors
            - delta: probability of theta in the confidence bound
            - alpha: tuning the exploration parameter
            - lambda_: regularization parameter
            - s: constant such that L2 norm of theta smaller than s
            - name: additional suffix when comparing several policies (optional)
            - sm: Should Sherman-Morisson formula be used for inverting matrices ?
            - sigma_noise: square root of the variance of the noise
            - verbose: To print information
            - omniscient: Does the policy knows when the breakpoints happen ?
        ACTION NORMS ARE SUPPOSED TO BE BOUNDED BE 1
        """
        # immediate attributes from the constructor
        self.name = name
        self.d = d
        self.sigma_noise = sigma_noise
        self.delta = delta
        self.lambda_ = lambda_
        self.action_norm_bound = action_norm_bound
        self.theta_norm_bound = theta_norm_bound
        self.color = color

    def init(self):
        # initialize calculators for LinUCB
        self.matrix = self.lambda_ * np.identity(self.d)
        self.b = np.zeros(self.d)
        self.t = 1

    def select_arm(self, arms):
        # calculate fixed parameters first
        # calculate estimated theta
        hat_theta = np.inner(np.linalg.inv(self.matrix), self.b)
        # calculate beta
        numerator_term = 1 + self.t * self.action_norm_bound ** 2 / self.lambda_
        log_term = np.log(numerator_term / self.delta)
        sqrt_term = np.sqrt(self.d * log_term)
        beta_term1 = self.sigma_noise * sqrt_term
        beta_term2 = np.sqrt(self.lambda_) * self.theta_norm_bound
        beta = beta_term1 + beta_term2
        # calculate inverse matrix
        inv_matrix = np.linalg.inv(self.matrix)

        # calculate ucbs
        kt = len(arms)
        ucbs = np.zeros(kt)
        for (i, arm) in enumerate(arms):
            estimated_reward = np.inner(arm.arm_feature, hat_theta)
            matrix_norm_term_one = np.dot(arm.arm_feature, inv_matrix)
            matrix_norm_term_two = np.inner(matrix_norm_term_one, arm.arm_feature)
            matrix_norm = np.sqrt(matrix_norm_term_two)
            ucbs[i] = estimated_reward + beta * matrix_norm

        # select arm
        mixer = np.random.random(ucbs.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucbs)))  # Sort the indices
        output = ucb_indices[::-1]  # Reverse list
        chosen_arm = output[0]
        return chosen_arm

    def update(self, chosen_arm, round_reward):
        # update calculators for LinUCB
        selected_arm_feature = chosen_arm.arm_feature
        matrix_addition = np.outer(selected_arm_feature, selected_arm_feature)
        self.matrix += matrix_addition
        b_addition = round_reward * selected_arm_feature
        self.b += b_addition
        self.t += 1