import numpy as np
from class_Arm import Arm


class Environment(object):
    def __init__(self, k, d, sigma_noise, variation_budget, time_horizon, stationary_degree):
        self.k = k
        self.d = d
        self.sigma_noise = sigma_noise
        self.variation_budget = variation_budget
        self.time_horizon = time_horizon
        self.stationary_degree = stationary_degree

        self.unif_min = -1
        self.unif_max = 1
        # self.re_init()

    def init(self):
        # self.expected_rewards = np.random.uniform(self.unif_min, self.unif_max, self.k)
        self.expected_rewards = np.zeros(self.k)
        self.expected_rewards1 = np.zeros(self.time_horizon)
        self.expected_rewards2 = np.zeros(self.time_horizon)
        self.t = 0
        sin_molecule = 5 * self.variation_budget * np.pi * self.t
        # change frequently
        first_sin_term = sin_molecule / (self.stationary_degree * self.time_horizon)
        # change gently
        # first_sin_term = sin_molecule / (3 * self.time_horizon)
        # change acutely
        second_sin_term = np.pi + sin_molecule / (self.stationary_degree * self.time_horizon)
        # change gently
        # second_sin_term = np.pi + sin_molecule / (3 * self.time_horizon)
        theta_first_term = 0.5 + 0.3 * np.sin(first_sin_term)
        theta_second_term = 0.5 + 0.3 * np.sin(second_sin_term)
        self.theta = np.array([theta_first_term, theta_second_term])
        self.arms = []
        for i in range(self.k):
            arm_id = i
            arm_feature = np.identity(self.d)[i]
            arm_expected_reward = arm_feature.dot(self.theta)
            self.expected_rewards[i] = arm_expected_reward
            self.arms.append(Arm(arm_id, arm_feature, arm_expected_reward))
        # print('self.t:', self.t)
        # print('self.theta:', self.theta)
        # print('self.expected_rewards:', self.expected_rewards)
        # for arm in self.arms:
        #     print('arm.arm_id：', arm.arm_id)
        #     print('arm.arm_feature：', arm.arm_feature)
        #     print('arm.arm_expected_reward：', arm.arm_expected_reward)
        # print()

    def re_init(self):
        self.t = 0

    def update(self):
        self.t += 1
        sin_molecule = 5 * self.variation_budget * np.pi * self.t
        # change acutely
        first_sin_term = sin_molecule / (self.stationary_degree * self.time_horizon)
        # change gently
        # first_sin_term = sin_molecule / (3 * self.time_horizon)
        # change acutely
        second_sin_term = np.pi + sin_molecule / (self.stationary_degree * self.time_horizon)
        # change gently
        # second_sin_term = np.pi + sin_molecule / (3 * self.time_horizon)

        theta_first_term = 0.5 + 0.3 * np.sin(first_sin_term)
        theta_second_term = 0.5 + 0.3 * np.sin(second_sin_term)
        self.theta = np.array([theta_first_term, theta_second_term])
        # print('theta:', self.theta)
        # print('norm:', np.linalg.norm(self.theta))
        for i in range(self.k):
            arm_id = i
            arm_feature = np.identity(self.d)[i]
            arm_expected_reward = arm_feature.dot(self.theta)
            self.expected_rewards[i] = arm_expected_reward
            self.arms[arm_id].arm_expected_reward = arm_expected_reward
        self.expected_rewards1[self.t - 1] = self.expected_rewards[0]
        self.expected_rewards2[self.t - 1] = self.expected_rewards[1]
        # print('self.t:', self.t)
        # print('self.theta:', self.theta)
        # print('self.expected_rewards:', self.expected_rewards)
        # for arm in self.arms:
        #     print('arm.arm_id：', arm.arm_id)
        #     print('arm.arm_feature：', arm.arm_feature)
        #     print('arm.arm_expected_reward：', arm.arm_expected_reward)
        # print()

    def play(self, choice):
        # print('arm_id:', self.arms[choice].arm_id)
        # print('arm_expected_reward:', self.arms[choice].arm_expected_reward)
        reward = self.arms[choice].pull(self.sigma_noise)
        return reward

    def get_expected_reward(self, choice):
        # print('self.expected_rewards:', self.expected_rewards)
        expected_reward = self.expected_rewards[choice]
        return expected_reward

    def get_optimal_expected_reward(self):
        optimal_expected_reward = np.max(self.expected_rewards)
        return optimal_expected_reward


# k = 3
# d = 3
# sigma_noise = 0.1
# environment = Environment(k, d, sigma_noise)
# environment.re_init()
