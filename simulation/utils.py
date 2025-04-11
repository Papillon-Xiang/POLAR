import csv
import os
from collections import deque
# from tqdm import tqdm
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
from POLAR import *


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
    seed = 0
    np.random.seed(seed)
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
    seed = 0
    np.random.seed(seed)
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

def standard_Q_learning(D, dim_h, L, dim_s_history):
    # offline data D: (s1, a1, s2, a2, s3, a3, s4)
    # D.shape = (n, dim_h[3])
    n = D.shape[0]
    s1= D[:,0:dim_h[0]]
    a1= D[:,dim_h[0]].astype(int)
    s2= D[:, (dim_h[0]+1):dim_h[1]]
    h2= D[:,0:dim_h[1]]
    a2= D[:,dim_h[1]].astype(int)
    s3= D[:, (dim_h[1]+1):dim_h[2]]
    h3= D[:,0:dim_h[2]]
    a3= D[:,dim_h[2]].astype(int)
    s4= D[:, (dim_h[2]+1):dim_h[3]]
    sh2 = np.concatenate( (s1, s2), axis=1 )   # 'state history'
    sh3 = np.concatenate( (s1, s2, s3), axis=1 )
    r1= r_bar(1,s1,a1,s2)
    r2= r_bar(2,h2,a2,s3)
    r3= r_bar(3,h3,a3,s4)
    
    design_matrix_1 = generate_design_matrix(s1)
    design_matrix_2 = generate_design_matrix(sh2)
    design_matrix_3 = generate_design_matrix(sh3)
    
    theta_1 = np.zeros((L*dim_s_history[0],2))
    theta_2 = np.zeros((L*dim_s_history[1],4))
    theta_3 = np.zeros((L*dim_s_history[2],8))
    
    stage1_missing = np.zeros(2)
    stage2_missing = np.zeros(4)
    stage3_missing = np.zeros(8)
    ##### idea:
    # if (act1, act2, act3) is not missing, then Q3(act1, act2, act3) is learned
    # if (act1, act2, act3) is missing, set theta3 s.t. Q3(act1, act2, act3) = -inf
    
    # compute theta3
    for act1 in range(2):   
        for act2 in range(2):
            for act3 in range(2):
                if np.sum((a1==act1)*(a2==act2)*(a3==act3))>0:
                    stage3_missing[act1*4+act2*2+act3] = 0   # not missing
                    theta_3[:, act1*4+act2*2+act3] = lstsq(design_matrix_3[(a1==act1)*(a2==act2)*(a3==act3),:],
                                                           r3[(a1==act1)*(a2==act2)*(a3==act3)])[0]
                else:
                    stage3_missing[act1*4+act2*2+act3] = 1   # missing
                    theta_3[:, act1*4+act2*2+act3] = -np.inf*np.ones(L*dim_s_history[2])   # set to -inf
                    
    # compute V3^hat
    Q3_hat_larger = (design_matrix_3 + 1e-100) @ theta_3
    Q3_hat_0 = Q3_hat_larger[np.arange(n), a1*4+a2*2]
    Q3_hat_1 = Q3_hat_larger[np.arange(n), a1*4+a2*2+1]
    V3_hat = np.maximum(Q3_hat_0, Q3_hat_1)
    
    # compute theta2
    for act1 in range(2):   
        for act2 in range(2):
            if np.sum((a1==act1)*(a2==act2))>0:
                stage2_missing[act1*2+act2] = 0   # not missing
                theta_2[:, act1*2+act2] = lstsq(design_matrix_2[(a1==act1)*(a2==act2),:],
                                                (r2+V3_hat)[(a1==act1)*(a2==act2)])[0]
            else:
                stage2_missing[act1*2+act2] = 1   # missing
                theta_2[:, act1*2+act2] = -np.inf*np.ones(L*dim_s_history[1])   # set to -inf
    
    # compute V2^hat
    Q2_hat_larger = (design_matrix_2 + 1e-100) @ theta_2
    Q2_hat_0 = Q2_hat_larger[np.arange(n), a1*2]
    Q2_hat_1 = Q2_hat_larger[np.arange(n), a1*2+1]
    V2_hat = np.maximum(Q2_hat_0, Q2_hat_1)
    
    # compute theta1
    for act1 in range(2):
        if np.sum(a1==act1)>0:
            stage1_missing[act1] = 0   # not missing
            theta_1[:, act1] = lstsq(design_matrix_1[(a1==act1),:], (r1+V2_hat)[(a1==act1)])[0]
        else:
            stage1_missing[act1] = 1   # missing
            theta_1[:, act1] = -np.inf*np.ones(L*dim_s_history[0])   # set to -inf

    return [theta_1, theta_2, theta_3]
