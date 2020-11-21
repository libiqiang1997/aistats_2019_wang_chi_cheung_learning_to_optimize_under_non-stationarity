import numpy as np


class SW_UCB(object):
    def __init__(self, d, w, k, sigma_noise, delta, lambda_, action_norm_bound, theta_norm_bound):
        self.d = d
        self.w = w
        self.k = k
        self.sigma_noise = sigma_noise
        self.delta = delta
        self.lambda_ = lambda_
        self.action_norm_bound = action_norm_bound
        self.theta_norm_bound = theta_norm_bound

        # self.re_init()

    def re_init(self):
        self.t = 1
        self.matrix = self.lambda_ * np.identity(self.d)
        # # self.b = np.zeros(self.d)
        self.b = np.zeros(self.d)
        self.feature_window = []
        self.reward_window = np.zeros(self.w)
        self.reward_cursor = 0
        # print('self.t:', self.t)
        # print('self.feature_window:', self.feature_window)
        # print('self.reward_window:', self.reward_window)

    def select_arm(self, arms):
        self.hat_theta = np.dot(np.linalg.inv(self.matrix), self.b)
        # print('hat_theta:', self.hat_theta)
        ucbs = np.zeros(self.k)
        for arm in arms:
            # estimated_reward = self.total_rewards[arm.arm_id] / self.chosen_counts[arm.arm_id]
            estimated_reward = np.dot(arm.arm_feature, self.hat_theta)

            denominator_term = 1 + self.w * self.action_norm_bound ** 2 / self.lambda_
            log_term = np.log(denominator_term / self.delta)
            sqrt_term = np.sqrt(self.d * log_term)
            beta_term1 = self.sigma_noise * sqrt_term
            beta_term2 = np.sqrt(self.lambda_) * self.theta_norm_bound
            beta = beta_term1 + beta_term2
            # beta = self.sigma_noise * np.sqrt(2 / self.chosen_counts[arm.arm_id] * np.log(2 * self.k * self.t / self.delta))
            inv_matrix = np.linalg.inv(self.matrix)
            matrix_norm_term1 = np.dot(arm.arm_feature, inv_matrix)
            matrix_norm_term2 = np.dot(matrix_norm_term1, arm.arm_feature)
            matrix_norm = np.sqrt(matrix_norm_term2)
            ucbs[arm.arm_id] = estimated_reward + beta * matrix_norm
            # print('arm_id:', arm.arm_id)
            # print('arm_feature:', arm.arm_feature)
            # print('arm_expected_reward:', arm.arm_expected_reward)
            # print('estimated_reward:', estimated_reward)
            # print('beta:', beta)
            # print('inv_matrix:', inv_matrix)
            # print('matrix_norm_term1:', matrix_norm_term1)
            # print('matrix_norm_term2:', matrix_norm_term2)
            # print('matrix_norm:', matrix_norm)
            # print('ucbs:', ucbs)
        # print('ucbs:', ucbs)
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
        # if self.t <= self.w:
        #     self.feature_window.append(selected_arm_feature)
        #     self.reward_window[self.reward_cursor] = round_reward
        #     self.reward_cursor += 1
        # print('self.t:', self.t)
        # print('self.feature_window:', self.feature_window)
        # print('self.reward_window:', self.reward_window)
        self.t += 1
        # print('self.b:', self.b)

    def __str__(self):
        return 'SW-UCB'

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
