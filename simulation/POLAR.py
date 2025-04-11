import csv
import os
from collections import deque
import copy

import pandas as pd
import os
import sys

from scipy.interpolate import BSpline
from scipy.linalg import lstsq
from joblib import Parallel, delayed
import itertools

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import chi2

def r_bar(k, k_max, h, a, s_new, dim_h):
    if k < k_max:
        return np.zeros(h.shape[0])

    if k == k_max:
        # Calculate reward only at the final step
        s_k_1, s_k_2 = h[:, dim_h[k - 2] + 1], h[:, dim_h[k - 2] + 2]
        s_next_1, s_next_2 = s_new[:, 0], s_new[:, 1]
        reward = (
            np.cos(-np.pi * s_k_1)
            + 2 * np.cos(np.pi * s_k_2)
            + s_next_1
            + 2 * s_next_2
        )
        return (reward - 1.37) * 3.8

    raise ValueError("Invalid value for k.")


def phi(k, h, a, dim_h):
    m = h.shape[0]
    if k == 1:
        s_k = h  # Initial state
    elif k > 1:
        s_k = h[:, dim_h[k - 2] + 1:]
    else:
        raise ValueError("Invalid value for k.")

    feature_matrix = np.concatenate(
        [
            (1 - a).reshape(m, 1),
            s_k * (1 - a)[:, np.newaxis],
            a.reshape(m, 1),
            s_k * a[:, np.newaxis],
        ],
        axis=1,
    )

    return feature_matrix


def r_true_expected(k, k_max, h, a, W_stars, dim_h):
    W_star = W_stars.get(k)
    if W_star is None:
        raise ValueError(f"Invalid value for k={k}, W_star not found.")

    expected_state = phi(k, h, a, dim_h) @ W_star.T

    return r_bar(k, k_max, h, a, expected_state, dim_h)


def get_transition(k, W, h, a, dim_h, dim_s):
    m = h.shape[0]
    mean = phi(k, h, a, dim_h) @ W.T   # (m, dim_s[k])
    noise = 0.8 * (np.random.beta(2, 2, size=(m, dim_s[k])) - 0.5)   # Beta(2,2), normalized to [-0.4, 0.4]
    new_state = np.minimum(np.maximum(mean + noise, 0), 1)
    return new_state


def compute_Q_values(k, h_tuple, k_max, W_stars, dim_h, dim_s):
    h = np.array(h_tuple)
    m = h.shape[0]

    if k == k_max:
        Q0 = r_true_expected(k, k_max, h, np.zeros(m), W_stars, dim_h)
        Q1 = r_true_expected(k, k_max, h, np.ones(m), W_stars, dim_h)
        return Q0, Q1

    def compute_Q(a):
        s_next = get_transition(k, W_stars[k], h, a, dim_h, dim_s)
        r = r_bar(k, k_max, h, a, s_next, dim_h)
        h_next = np.concatenate((h, a.reshape(m, 1), s_next), axis=1)
        V_next, _ = compute_V_Pi_star(k + 1, tuple(map(tuple, h_next)), k_max, W_stars, dim_h, dim_s)
        return r + V_next

    Q0 = compute_Q(np.zeros(m))
    Q1 = compute_Q(np.ones(m))
    return Q0, Q1


def compute_V_Pi_star(k, h_tuple, k_max, W_stars, dim_h, dim_s):
    Q0, Q1 = compute_Q_values(k, h_tuple, k_max, W_stars, dim_h, dim_s)
    V = np.maximum(Q0, Q1)
    Pi = (Q0 < Q1).astype(int)
    return V, Pi


def get_optimal_policy_functions_new(k_max, W_stars, dim_h, dim_s):
    def make_policy_function(k):
        def policy_function(h):
            h_tuple = tuple(map(tuple, h))
            _, Pi = compute_V_Pi_star(k, h_tuple, k_max, W_stars, dim_h, dim_s)
            return Pi
        return policy_function

    return (make_policy_function(k) for k in range(1, k_max + 1))


def generate_offline_data(num_samples, p, k_max, W_stars, dim_h, dim_s):
    policy_functions = list(get_optimal_policy_functions_new(k_max, W_stars, dim_h, dim_s))

    histories = [np.random.uniform(0, 1, size=(num_samples, dim_s[0]))]
    actions = []

    for k, Pi_star in enumerate(policy_functions, start=1):
        h = histories[-1]
        a_optimal = Pi_star(h)
        u = np.random.binomial(1, p, size=num_samples)
        a = ((2 * a_optimal - 1) * (2 * u - 1) + 1) / 2
        actions.append(a)

        if k < k_max:
            s_next = get_transition(k, W_stars[k], h, a, dim_h, dim_s)
            h_next = np.concatenate((h, a.reshape(num_samples, 1), s_next), axis=1)
            histories.append(h_next)

    final_states = get_transition(k_max, W_stars[k_max], histories[-1], actions[-1], dim_h, dim_s)
    final_data = np.concatenate((histories[-1], actions[-1].reshape(num_samples, 1), final_states), axis=1)

    return final_data



def get_Lamda(k, D, ridge_para, dim_s, dim_h):
    n = D.shape[0]
    d_phi_k = 2 * (1 + dim_s[k - 1])
    Lamda = ridge_para * np.eye(d_phi_k)

    phi_matrix = phi(k, D[:, :dim_h[k - 1]], D[:, dim_h[k - 1]], dim_h)

    Lamda += np.sum(
        phi_matrix[:, :, np.newaxis] * phi_matrix[:, np.newaxis, :], axis=0
    )

    return Lamda


def get_Gamma(k, h, a, D, ridge_para, dim_s, dim_h):
    Lamda = get_Lamda(k, D, ridge_para, dim_s, dim_h)  # (d_phi_k, d_phi_k)
    phi_matrix = phi(k, h, a, dim_h)  # (m, d_phi_k)
    Gamma = np.sqrt(
        np.sum((phi_matrix @ np.linalg.inv(Lamda)) * phi_matrix, axis=1)
    )
    return Gamma


def estimate_W(k, D, ridge_para, dim_s, dim_h):
    n = D.shape[0]
    d_phi_k = 2 * (1 + dim_s[k - 1])

    Lamda = get_Lamda(k, D, ridge_para, dim_s, dim_h)

    S_next_matrix = D[:, (dim_h[k - 1] + 1) : dim_h[k]]
    phi_matrix = phi(k, D[:, :dim_h[k - 1]], D[:, dim_h[k - 1]], dim_h)

    A = S_next_matrix.T @ phi_matrix

    W_hat = A @ np.linalg.inv(Lamda)
    return W_hat


def generate_design_matrix(sh, knots, L, k_degree):
    m, sh_length = sh.shape[0], sh.shape[1]
    design_matrix = np.concatenate(
        [
            np.array(
                [
                    BSpline(t=knots, c=np.eye(1, L, i)[0], k=k_degree)(sh[:, j])
                    for i in range(L)
                ]
            ).T
            for j in np.arange(sh_length)
        ],
        axis=1,
    )
    return design_matrix


def get_action(k, theta, h, dim_h, knots, L, k_degree, greedy=False):
    m = h.shape[0]

    if k == 1:
        s_history = h
    else:
        action_col_indices = dim_h[: k - 1]
        action_index = np.zeros(m, dtype=int)
        for col_idx in action_col_indices:
            a_col = h[:, col_idx].astype(int)
            action_index = 2 * action_index + a_col

        s_history = np.delete(h, action_col_indices, axis=1)

    b = generate_design_matrix(s_history, knots, L, k_degree)
    f_larger = (b + 1e-100) @ theta

    if k == 1:
        f = f_larger
    else:
        row_index = np.array([np.arange(m), np.arange(m)]).T.astype(int)
        col_index = np.array([2 * action_index, 2 * action_index + 1]).T.astype(int)
        f = f_larger[row_index, col_index]

    exp_f = np.exp(f - np.max(f, axis=1)[:, np.newaxis])
    prob = exp_f / np.sum(exp_f, axis=1)[:, np.newaxis]
    u = np.random.uniform(0, 1, size=m)
    a = np.argmax(np.cumsum(prob, axis=1) > u.reshape(-1, 1), axis=1)

    if greedy:
        a = (f[:, 1] > f[:, 0]).astype(int)

    return a



def compute_V1_unif(W_stars, theta_list, n_MC, k_max, dim_s, dim_h, knots, L, k_degree, greedy=False):

    h = np.random.uniform(0, 1, size=(n_MC, dim_s[0]))
    V = 0

    for k in range(1, k_max + 1):
        theta = theta_list[k - 1]
        a = get_action(k, theta, h, dim_h, knots, L, k_degree, greedy)
        r = r_true_expected(k, k_max, h, a, W_stars, dim_h)
        V += r

        if k < k_max:
            s_next = get_transition(k, W_stars[k], h, a, dim_h, dim_s)
            h = np.concatenate((h, a.reshape(n_MC, 1), s_next), axis=1)

    return np.mean(V)

def generate_sample_for_regression(m, k_max, dim_s, knots, L, k_degree):

    s_list = [np.random.uniform(0, 1, size=(m, dim_s[k - 1])) for k in range(1, k_max + 1)]
    design_matrices = [generate_design_matrix(np.concatenate(s_list[:k], axis=1), knots, L, k_degree) for k in range(1, k_max + 1)]
    return s_list, design_matrices


def r_hat(k, k_max, h, a, W_hat_list, dim_h):
    if not (1 <= k <= len(W_hat_list)):
        raise ValueError(f"Invalid value for k={k}. Ensure 1 <= k <= len(W_hat_list).")

    W_hat = W_hat_list[k - 1].T
    expected_state = phi(k, h, a, dim_h) @ W_hat
    return r_bar(k, k_max, h, a, expected_state, dim_h)


def r_tilde(k, k_max, h, dim_h, dim_s, a, W_hat_list, D_offline, ridge_para, c_pessimism):
    expected_reward = r_hat(k, k_max, h, a, W_hat_list, dim_h)
    uncertainty_penalty = c_pessimism * get_Gamma(k, h, a, D_offline, ridge_para, dim_s, dim_h)
    return expected_reward - uncertainty_penalty




def compute_Q_function(k, k_degree, h, dim_h, dim_s, knots, L, a, W_hat_list, theta_list, r_function, n_MC, k_max, D_offline, ridge_para, c_pessimism):
    m = h.shape[0]

    # Base case: if at maximum depth, compute the immediate reward
    if k == k_max:
        Q_k = r_function(k, k_max, h, dim_h, dim_s, a, W_hat_list, D_offline, ridge_para, c_pessimism)
        return Q_k

    # Compute immediate reward
    r_k = r_function(k, k_max, h, dim_h, dim_s, a, W_hat_list, D_offline, ridge_para, c_pessimism)

    # Monte Carlo sampling: repeat states and actions
    h_sample = np.repeat(h, n_MC, axis=0)
    a_sample = np.repeat(a, n_MC)
    s_next_sample = get_transition(k, W_hat_list[k - 1], h_sample, a_sample, dim_h, dim_s)
    # h_next_sample = np.concatenate((h_sample, a_sample.reshape(m * n_MC, 1), s_next_sample), axis=1)
    h_next_sample = np.concatenate(
        [h_sample, a_sample.reshape(-1, 1), s_next_sample],
        axis=1
    )
    
    a_next_sample = get_action(k + 1, theta_list[k], h_next_sample, dim_h, knots, L, k_degree)

    # Recursive computation of Q-value for the next step
    Q_next_sample = compute_Q_function(
        k + 1, k_degree, h_next_sample, dim_h, dim_s, knots, L, a_next_sample,
        W_hat_list, theta_list, r_function, n_MC=1, k_max=k_max,
        D_offline=D_offline, ridge_para=ridge_para, c_pessimism=c_pessimism
    )

    # Average over Monte Carlo samples
    Q_k = r_k + np.mean(Q_next_sample.reshape(m, n_MC), axis=1)
    return Q_k

def process_theta(k, k_degree, dim_h, dim_s, knots, L, actions, s_list, design_matrix, theta_list, W_hat_list, r_tilde, n_MC, eta, m, k_max, D_offline, ridge_para, c_pessimism):
    try:
        h = np.concatenate(
            [s_list[i // 2] if i % 2 == 0 else np.repeat(actions[i // 2], m).reshape(m, 1)
                for i in range(2 * k - 1)],
            axis=1,
        )

        Q_list  = compute_Q_function(
            k, k_degree, h, dim_h, dim_s, knots, L, np.repeat(actions[-1], m),
            W_hat_list, theta_list, r_tilde,
            n_MC, k_max, D_offline, ridge_para, c_pessimism
        )

        theta_increment = lstsq(design_matrix, Q_list)[0]
        action_index = sum(actions[i] * (2 ** (k - 1 - i)) for i in range(len(actions)))
        theta_new = theta_list[k - 1][:, action_index] + eta * theta_increment
        return actions, theta_new
    except Exception as e:
        print(f"Error in process_theta: {str(e)}")
        raise

def one_iteration_parallel(dim_s, dim_h, k_degree, knots, L, theta_list, W_hat_list, eta, n_MC, m, r_tilde, k_max, D_offline, ridge_para, c_pessimism):
    try:
        s_list, design_matrices = generate_sample_for_regression(m, k_max, dim_s, knots, L, k_degree)
        theta_list_new = [np.zeros(theta.shape) for theta in theta_list]

        for k in range(1, k_max + 1):
            results = Parallel(n_jobs=-1)(
                delayed(process_theta)(
                    k, k_degree, dim_h, dim_s, knots, L, actions, s_list[:k], design_matrices[k - 1],
                    theta_list, W_hat_list, r_tilde, n_MC, eta, m,
                    k_max, D_offline, ridge_para, c_pessimism
                )
                for actions in np.ndindex(*(2,) * k)
            )

            for actions, theta_new in results:
                action_index = sum(actions[i] * (2 ** (k - 1 - i)) for i in range(len(actions)))
                theta_list_new[k - 1][:, action_index] = theta_new

        return theta_list_new
    except Exception as e:
        print(f"Error in one_iteration_parallel: {str(e)}")
        raise


