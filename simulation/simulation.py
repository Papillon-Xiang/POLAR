import numpy as np

dim_s = np.array([2,2,2,2])
dim_s_history = np.cumsum(dim_s)
dim_h = dim_s_history + np.arange(4)
d_phi_1, d_phi_2, d_phi_3 = 2*(1+dim_s[0]), 2*(1+dim_s[1]), 2*(1+dim_s[2])

k_degree= 3
m_interior= 3
knots= np.concatenate( (np.zeros(k_degree), np.linspace(0,1, m_interior+2), np.ones(k_degree)) )
L = k_degree + m_interior + 1


W1_star = np.array([[0.4, 0.2, 0, 0.6, 0, -0.2],
                    [0.4, 0, 0.2, 0.4, 0.2, 0]])
W2_star = np.array([[0.5, 0.1, -0.1, 0.5, -0.1, 0.1],
                    [0.5, -0.1, 0.1, 0.5, 0.1, -0.1]])
W3_star = np.array([[0.6, -0.12, -0.08, 0.4, 0.08, 0.12],
                    [0.6, -0.08, -0.12, 0.4, 0.12, 0.08]])

def phi(k, h, a):

    m = h.shape[0]

    if k == 1:
        s = h
    elif k == 2:
        s = h[:, dim_h[0] + 1:]
    elif k == 3:
        s = h[:, dim_h[1] + 1:]
    else:
        raise ValueError("Invalid value for k.")

    feature_matrix = np.concatenate([
        (1 - a).reshape(m, 1),
        s * (1 - a)[:, np.newaxis],
        a.reshape(m, 1),
        s * a[:, np.newaxis]
    ], axis=1)

    return feature_matrix


def r_bar(k, h, a, s_new):

    if k in [1, 2]:
        return np.zeros(h.shape[0])

    if k == 3:
        s3_1, s3_2 = h[:, dim_h[1] + 1], h[:, dim_h[1] + 2]
        s4_1, s4_2 = s_new[:, 0], s_new[:, 1]
        reward = (np.cos(-np.pi * s3_1) + 2 * np.cos(np.pi * s3_2) + s4_1 + 2 * s4_2)
        return (reward - 1.37) * 3.8

    raise ValueError("Invalid value for k.")

def r_true_expected(k, h, a):

    W_star = {1: W1_star, 2: W2_star, 3: W3_star}.get(k)
    if W_star is None:
        raise ValueError("Invalid value for k.")

    expected_state = phi(k, h, a) @ W_star.T
    return r_bar(k, h, a, expected_state)

def get_transition(k, W, h, a):

    m = h.shape[0]
    mean = phi(k,h,a) @ W.T   # (m, dim_s[k])
    noise = 0.8* (np.random.beta(2,2, size=(m, dim_s[k])) - 0.5)   # Beta(2,2), normalized to [-0.4, 0.4], shape is (m,)
    new_state = np.minimum( np.maximum( mean+noise, 0), 1)
    return new_state

from functools import lru_cache
@lru_cache(maxsize=None)

def compute_Q_values(k, h_tuple):
    h = np.array(h_tuple)
    m = h.shape[0]

    if k == 3:
        Q0 = r_true_expected(k, h, np.zeros(m))
        Q1 = r_true_expected(k, h, np.ones(m))
        return Q0, Q1

    elif k == 2:
        def compute_Q2(a2):
            s3 = get_transition(2, W2_star, h, a2)
            r2 = r_bar(2, h, a2, s3)
            h3 = np.concatenate((h, a2.reshape(m, 1), s3), axis=1)
            V3, _ = compute_V_Pi_star(3, tuple(map(tuple, h3)))
            return r2 + V3

        Q0 = compute_Q2(np.zeros(m))
        Q1 = compute_Q2(np.ones(m))
        return Q0, Q1

    elif k == 1:
        def compute_Q1(a1):
            s2 = get_transition(1, W1_star, h, a1)
            r1 = r_bar(1, h, a1, s2)
            h2 = np.concatenate((h, a1.reshape(m, 1), s2), axis=1)
            V2, _ = compute_V_Pi_star(2, tuple(map(tuple, h2)))
            return r1 + V2

        Q0 = compute_Q1(np.zeros(m))
        Q1 = compute_Q1(np.ones(m))
        return Q0, Q1

def compute_V_Pi_star(k, h_tuple):
    Q0, Q1 = compute_Q_values(k, h_tuple)
    V = np.maximum(Q0, Q1)
    Pi = (Q0 < Q1).astype(int)
    return V, Pi

def get_optimal_policy_functions_new():
    def make_policy_function(k):
        def policy_function(h):
            h_tuple = tuple(map(tuple, h))
            _, Pi = compute_V_Pi_star(k, h_tuple)
            return Pi
        return policy_function

    return (make_policy_function(k) for k in (1, 2, 3))


def generate_full_offline_data(num_samples):
    Pi_1_star, Pi_2_star, Pi_3_star = get_optimal_policy_functions_new()

    s1 = np.random.uniform(0, 1, size=(num_samples, dim_s[0]))
    a1_optimal = Pi_1_star(s1)

    s2 = get_transition(1, W1_star, s1, a1_optimal)
    h2 = np.concatenate((s1, a1_optimal.reshape(num_samples, 1), s2), axis=1)
    a2_optimal = Pi_2_star(h2)

    s3 = get_transition(2, W2_star, h2, a2_optimal)
    h3 = np.concatenate((h2, a2_optimal.reshape(num_samples, 1), s3), axis=1)
    a3_optimal = Pi_3_star(h3)
    s4 = get_transition(3, W3_star, h3, a3_optimal)

    return np.concatenate((h3, a3_optimal.reshape(num_samples, 1), s4), axis=1)

def optimal_action(num_samples, Pi_1_star, Pi_2_star, Pi_3_star):
    s1 = np.random.uniform(0, 1, size=(num_samples, dim_s[0]))
    a1_optimal = Pi_1_star(s1)

    s2 = get_transition(1, W1_star, s1, a1_optimal)
    h2 = np.concatenate((s1, a1_optimal.reshape(num_samples, 1), s2), axis=1)

    a2_optimal = Pi_2_star(h2)

    s3 = get_transition(2, W2_star, h2, a2_optimal)
    h3 = np.concatenate((h2, a2_optimal.reshape(num_samples, 1), s3), axis=1)

    a3_optimal = Pi_3_star(h3)
    s4 = get_transition(3, W3_star, h3, a3_optimal)

    full_offline_data = np.concatenate((h3, a3_optimal.reshape(num_samples, 1), s4), axis=1)
    return a1_optimal, a2_optimal, a3_optimal, s1, h2, h3