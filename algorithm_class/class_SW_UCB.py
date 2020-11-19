import numpy as np


class SW_UCB(object):
    def __init__(self, d, sigma_noise, k, delta, lambda_):
        self.d = d
        self.sigma_noise = sigma_noise
        self.k = k
        self.delta = delta
        self.lambda_ = lambda_

        # self.re_init()

    def re_init(self):
        self.t = 1
        self.matrix = self.lambda_ * np.identity(self.d)
        # # self.b = np.zeros(self.d)
        self.b = np.zeros(self.d)
        # self.total_rewards = np.zeros(self.k)
        # self.chosen_counts = np.zeros(self.k)
        # self.ucbs = np.ones(self.k) * float('inf')

    def select_arm(self, arms):
        self.hat_theta = np.dot(np.linalg.inv(self.matrix), self.b)
        # print('hat_theta:', self.hat_theta)
        ucbs = np.zeros(self.k)
        for arm in arms:
            # estimated_reward = self.total_rewards[arm.arm_id] / self.chosen_counts[arm.arm_id]
            estimated_reward = np.dot(arm.arm_feature, self.hat_theta)
            alpha = self.sigma_noise * np.sqrt(2 * np.log(1 / self.delta)) + self.lambda_ * np.sqrt(self.d)
            # alpha = self.sigma_noise * np.sqrt(2 / self.chosen_counts[arm.arm_id] * np.log(2 * self.k * self.t / self.delta))
            inv_matrix = np.linalg.inv(self.matrix)
            matrix_norm_temp = np.dot(arm.arm_feature, inv_matrix)
            matrix_norm_temp1 = np.dot(matrix_norm_temp, arm.arm_feature)
            matrix_norm = np.sqrt(matrix_norm_temp1)
            ucbs[arm.arm_id] = estimated_reward + alpha * matrix_norm
            # print('arm_id:', arm.arm_id)
            # print('arm_feature:', arm.arm_feature)
            # print('arm_expected_reward:', arm.arm_expected_reward)
            # print('estimated_reward:', estimated_reward)
            # print('alpha:', alpha)
            # print('inv_matrix:', inv_matrix)
            # print('matrix_norm_temp:', matrix_norm_temp)
            # print('matrix_norm_temp1:', matrix_norm_temp1)
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
        self.t += 1
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
