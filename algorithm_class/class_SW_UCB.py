import numpy as np


class SW_UCB(object):
    def __init__(self, name, d, w, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound, color):
        self.name = name
        self.d = d
        self.w = w
        self.sigma_noise = sigma_noise
        self.delta = delta
        self.lambda_ = lambda_
        self.action_norm_bound = action_norm_bound
        self.theta_norm_bound = theta_norm_bound
        self.color = color

        # self.re_init()

    def init(self):
        # initialize calculators for SW-UCB
        self.matrix = self.lambda_ * np.identity(self.d)
        self.b = np.zeros(self.d)
        self.feature_window = []
        self.reward_window = np.zeros(self.w)
        # initialize time
        self.t = 1

    def select_arm(self, arms):
        # calculate fixed parameters first
        # calculate estimated theta
        hat_theta = np.inner(np.linalg.inv(self.matrix), self.b)
        # calculate beta
        numerator_term = 1 + self.w * self.action_norm_bound ** 2 / self.lambda_
        log_term = np.log(numerator_term / self.delta)
        sqrt_term = np.sqrt(self.d * log_term)
        beta_term1 = self.sigma_noise * sqrt_term
        beta_term2 = np.sqrt(self.lambda_) * self.theta_norm_bound
        beta = beta_term1 + beta_term2
        # calculate inverse matrix
        inv_matrix = np.linalg.inv(self.matrix)

        # initialize ucb array
        kt = len(arms)  # available actions at time t
        ucbs = np.zeros(kt)  # upper-confidence bounds for every action
        for (i, arm) in enumerate(arms):
            estimate_reward = np.inner(arm.arm_feature, hat_theta)
            matrix_norm_term_one = np.dot(arm.arm_feature, inv_matrix)
            matrix_norm_term_two = np.inner(matrix_norm_term_one, arm.arm_feature)
            matrix_norm = np.sqrt(matrix_norm_term_two)
            ucbs[i] = estimate_reward + beta * matrix_norm

        # select arm
        mixer = np.random.random(ucbs.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucbs)))  # Sort the indices
        output = ucb_indices[::-1]  # Reverse list
        chosen_arm = output[0]
        return chosen_arm

    def update(self, chosen_arm, round_reward):
        # update calculators for SW-UCB
        selected_arm_feature = chosen_arm.arm_feature
        matrix_addition = np.outer(selected_arm_feature, selected_arm_feature)
        self.matrix += matrix_addition
        b_addition = round_reward * selected_arm_feature
        self.b += b_addition

        # update window
        self.feature_window.append(selected_arm_feature)
        if self.t <= self.w:
            self.reward_window[self.t - 1] = round_reward
        elif self.t > self.w:
            # update window
            deducted_arm_feature = self.feature_window.pop(0)
            deducted_reward = self.reward_window[0]
            self.reward_window = np.roll(self.reward_window, -1)
            self.reward_window[len(self.reward_window) - 1] = round_reward
            # update model for calculation
            matrix_deduction = np.outer(deducted_arm_feature, deducted_arm_feature)
            self.matrix -= matrix_deduction
            b_deduction = deducted_reward * deducted_arm_feature
            self.b -= b_deduction

        # update time
        self.t += 1