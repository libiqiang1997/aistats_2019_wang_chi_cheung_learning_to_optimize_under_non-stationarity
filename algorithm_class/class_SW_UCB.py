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
        self.t = 1
        self.matrix = self.lambda_ * np.identity(self.d)
        # # self.b = np.zeros(self.d)
        self.b = np.zeros(self.d)
        self.feature_window = []
        self.reward_window = np.zeros(self.w)
        # print('self.t:', self.t)
        # print('self.feature_window:', self.feature_window)
        # print('self.reward_window:', self.reward_window)

    def select_arm(self, arms):
        kt = len(arms)  # available actions at time t
        ucbs = np.zeros(kt)  # upper-confidence bounds for every action

        hat_theta = np.inner(np.linalg.inv(self.matrix), self.b)
        # print('hat_theta:', self.hat_theta)

        denominator_term = 1 + self.w * self.action_norm_bound ** 2 / self.lambda_
        log_term = np.log(denominator_term / self.delta)
        sqrt_term = np.sqrt(self.d * log_term)
        beta_term1 = self.sigma_noise * sqrt_term
        beta_term2 = np.sqrt(self.lambda_) * self.theta_norm_bound
        beta = beta_term1 + beta_term2

        inv_matrix = np.linalg.inv(self.matrix)

        for (i, arm) in enumerate(arms):
            estimate_reward = np.inner(arm.arm_feature, hat_theta)
            matrix_norm_term1 = np.dot(arm.arm_feature, inv_matrix)
            matrix_norm_term2 = np.inner(matrix_norm_term1, arm.arm_feature)
            matrix_norm = np.sqrt(matrix_norm_term2)
            ucbs[i] = estimate_reward + beta * matrix_norm
        mixer = np.random.random(ucbs.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucbs)))  # Sort the indices
        output = ucb_indices[::-1]  # Reverse list
        chosen_arm = output[0]
        # print('ucbs:', self.ucbs)
        # print('mixer:', mixer)
        # print('ucb_indices:', ucb_indices)
        # print('output:', output)
        # print('chosen_arm:', chosen_arm)
        return chosen_arm

    def update(self, selected_arm_feature, round_reward):
        # # print('self.matrix:\n', self.matrix)
        # # print('selected_arm_feature:', selected_arm_feature)
        # # print('round_reward:', round_reward)
        matrix_addition = np.outer(selected_arm_feature, selected_arm_feature)
        # # print('matrix_addition:', matrix_addition)
        self.matrix += matrix_addition
        # # print('self.matrix:\n', self.matrix)
        b_addition = round_reward * selected_arm_feature
        # # print('self.b:', self.b)
        self.b += b_addition
        # self.chosen_counts[choice] += 1
        # self.total_rewards[choice] += round_reward
        self.feature_window.append(selected_arm_feature)
        if self.t <= self.w:
            self.reward_window[self.t - 1] = round_reward
        elif self.t > self.w:
            deducted_arm_feature = self.feature_window.pop(0)
            deducted_reward = self.reward_window[0]
            self.reward_window = np.roll(self.reward_window, -1)
            self.reward_window[len(self.reward_window) - 1] = round_reward
            matrix_deduction = np.outer(deducted_arm_feature, deducted_arm_feature)
            self.matrix -= matrix_deduction
            b_deduction = deducted_reward * deducted_arm_feature
            self.b -= b_deduction
        self.t += 1
        # if self.t <= self.w:
        #     self.feature_window.append(selected_arm_feature)
        #     self.reward_window[self.reward_cursor] = round_reward
        #     self.reward_cursor += 1
        # print('self.t:', self.t)
        # print('self.feature_window:', self.feature_window)
        # print('self.reward_window:', self.reward_window)
        # print('self.b:', self.b)


# k = 10
# # d = 5
# delta = 0.1
# sigma_noise = 1
# ucb = UCB(sigma_noise, k, delta)
# ucb.re_init()
# chosen_arm
# from class_Arm import Arm
# arm0 = Arm(0, 0.9)
# arm1 = Arm(1, 0.8)
# arm2 = Arm(2, 0.8)
# arms = [arm0, arm1, arm2]
# ucb = UCB(0.1, 3, 0.1)
# ucb.re_init()
# ucb.select_arm(arms)
