import numpy as np


class BOB(object):
    def __init__(self, name, d, sigma_noise, lambda_, action_norm_bound, theta_norm_bound, time_horizon, color):
        self.name = name
        self.d = d
        self.sigma_noise = sigma_noise
        self.lambda_ = lambda_
        self.action_norm_bound = action_norm_bound
        self.theta_norm_bound = theta_norm_bound
        self.time_horizon = time_horizon
        self.color = color

    def init(self):
        # initialize parameters for BOB
        # initialize parameters to determine window length w_i
        self.block_size = int(self.d ** (2 / 3) * self.time_horizon ** (1 / 2))
        self.num_block = self.upper_int(self.time_horizon / self.block_size)
        self.block_init_times = [i * self.block_size for i in range(self.num_block)]
        self.delta = self.upper_int(np.log(self.block_size))
        numerator_term = (self.delta + 1) * np.log(self.delta + 1)
        denominator_term = (np.e - 1) * self.num_block
        self.gamma = np.sqrt(numerator_term / denominator_term)
        if self.gamma > 1:
            self.gamma = 1
        self.weights = np.zeros(self.delta + 1)
        for j in range(len(self.weights)):
            self.weights[j] = 1

        # determine window length w_i
        self.w_i = self.determine_window_length()

        # initialize calculators for SW-UCB
        self.matrix = self.lambda_ * np.identity(self.d)
        self.b = np.zeros(self.d)
        self.feature_window = []
        self.reward_window = np.zeros(self.w_i)

        # initialize current block id
        self.current_block_id = 0
        # initialize block sum reward for EXP3
        self.block_sum_reward = 0
        # initialize time
        self.t = 1

    def select_arm(self, arms):
        # calculate fixed parameters first
        # calculate estimated theta
        hat_theta = np.inner(np.linalg.inv(self.matrix), self.b)
        # calculate beta
        numerator_term = 1 + self.w_i * self.action_norm_bound ** 2 / self.lambda_
        log_term = np.log(self.time_horizon * numerator_term)
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

        # update block sum reward for EXP3
        self.block_sum_reward += round_reward

        # update window for block
        self.feature_window.append(selected_arm_feature)
        current_block_t = self.t - self.current_block_id * self.block_size
        if current_block_t <= self.w_i:
            self.reward_window[current_block_t-1] = round_reward
        elif current_block_t > self.w_i:
            # update window
            deducted_arm_feature = self.feature_window.pop(0)
            deducted_reward = self.reward_window[0]
            self.reward_window = np.roll(self.reward_window, -1)
            self.reward_window[len(self.reward_window)-1] = round_reward
            # update model for calculation
            matrix_deduction = np.outer(deducted_arm_feature, deducted_arm_feature)
            self.matrix -= matrix_deduction
            b_deduction = deducted_reward * deducted_arm_feature
            self.b -= b_deduction

        # initialize for next block
        if self.current_block_id + 1 < self.num_block \
                and self.t == self.block_init_times[self.current_block_id + 1]:
            # update current block id
            self.current_block_id += 1

            # update weights
            first_multiplier_term = self.gamma / ((self.delta + 1) * self.probabilities[self.j_i])
            second_multiplier_sqrt_term = np.sqrt(
                self.block_size * np.log(self.time_horizon / np.sqrt(self.block_size)))
            second_multiplier_denominator_term = 2 * self.block_size + 4 * self.sigma_noise * second_multiplier_sqrt_term
            second_multiplier_term = 1 / 2 + self.block_sum_reward / second_multiplier_denominator_term
            exponentiation_term = np.e ** (first_multiplier_term * second_multiplier_term)
            self.weights[self.j_i] *= exponentiation_term

            # initialize block sum reward for EXP3
            self.block_sum_reward = 0

            # determine window length w_i
            self.w_i = self.determine_window_length()

            # initialize calculators for SW-UCB
            self.matrix = self.lambda_ * np.identity(self.d)
            self.b = np.zeros(self.d)
            self.feature_window = []
            self.reward_window = np.zeros(self.w_i)

        # update time
        self.t += 1

    def determine_window_length(self):
        # determine window length w_i
        self.j_i = -1
        sum_weight_term = np.sum(self.weights)
        self.probabilities = np.zeros(len(self.weights))
        for j in range(len(self.weights)):
            frac_term = self.weights[j] / sum_weight_term
            first_probability_term = (1 - self.gamma) * frac_term
            second_probability_term = self.gamma / (self.delta + 1)
            probability = first_probability_term + second_probability_term
            self.probabilities[j] = probability
        cum_probabilities = np.cumsum(self.probabilities)
        prop_rand = np.random.random()
        for j in range(len(cum_probabilities)):
            if prop_rand <= cum_probabilities[j]:
                self.j_i = j
                break
        w_i_exponent_term = self.j_i / self.delta
        w_i_exponentiation_term = self.block_size ** w_i_exponent_term
        w_i = int(w_i_exponentiation_term)
        return w_i

    def upper_int(self, float_num):
        if not float.is_integer(float_num):
            float_num += 1
        float_num = int(float_num)
        return float_num


