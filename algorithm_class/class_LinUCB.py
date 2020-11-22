#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Implementation of the classic LinUCB model
"""

# Author: Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import numpy as np
from math import log
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
        """
        Re-init function to reinitialize the statistics while keeping the same hyper-parameters
        """
        self.t = 1
        # self.hat_theta = np.zeros(self.d)
        self.matrix = self.lambda_ * np.identity(self.d)
        # self.invcov = 1 / self.lambda_ * np.identity(self.d)
        self.b = np.zeros(self.d)

    def select_arm(self, arms):
        """
        Selecting an arm according to the LinUCB policy
        param:
            - arms : list of objects Arm with contextualized features
        Output:
        -------
        chosen_arm : index of the pulled arm
        """
        # assert type(arms) == list, 'List of arms as input required'
        kt = len(arms)  # available actions at time t
        ucbs = np.zeros(kt)  # upper-confidence bounds for every action

        hat_theta = np.inner(np.linalg.inv(self.matrix), self.b)

        denominator_term = 1 + self.t * self.action_norm_bound ** 2 / self.lambda_
        log_term = np.log(denominator_term / self.delta)
        sqrt_term = np.sqrt(self.d * log_term)
        beta_term1 = self.sigma_noise * sqrt_term
        beta_term2 = np.sqrt(self.lambda_) * self.theta_norm_bound
        beta = beta_term1 + beta_term2

        inv_matrix = np.linalg.inv(self.matrix)

        for (i, arm) in enumerate(arms):
            estimated_reward = np.inner(arm.arm_feature, hat_theta)
            matrix_norm_term1 = np.dot(arm.arm_feature, inv_matrix)
            matrix_norm_term2 = np.inner(matrix_norm_term1, arm.arm_feature)
            matrix_norm = np.sqrt(matrix_norm_term2)
            ucbs[i] = estimated_reward + beta * matrix_norm
        mixer = np.random.random(ucbs.size)  # Shuffle to avoid always pulling the same arm when ties
        ucb_indices = list(np.lexsort((mixer, ucbs)))  # Sort the indices
        output = ucb_indices[::-1]  # Reverse list
        chosen_arm = output[0]
        return chosen_arm

    def update(self, selected_arm_feature, round_reward):
        """
        Updating the main parameters for the model
        param:
            - selected_arm_feature: Feature used for updating
            - round_reward: Reward used for updating
        Output:
        -------
        Nothing, but the class instances are updated
        """
        matrix_addition = np.outer(selected_arm_feature, selected_arm_feature)
        self.matrix += matrix_addition
        b_addition = round_reward * selected_arm_feature
        self.b += b_addition
        self.t += 1
        # self.hat_theta = np.inner(self.invcov, self.b)


    # @staticmethod
    # def id():
    #     return 'LinUCB'

