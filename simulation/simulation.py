import numpy as np
from tqdm import tqdm  
import time
from POLAR import *

def standard_Q_learning(D, k_max, dim_h, L, dim_s_history, knots, k_degree):
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
    r1= r_bar(1, k_max, s1,a1,s2, dim_h)
    r2= r_bar(2, k_max, h2,a2,s3, dim_h)
    r3= r_bar(3, k_max, h3,a3,s4, dim_h)
    
    design_matrix_1 = generate_design_matrix(s1, knots, L, k_degree)
    design_matrix_2 = generate_design_matrix(sh2, knots, L, k_degree)
    design_matrix_3 = generate_design_matrix(sh3, knots, L, k_degree)
    
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


if __name__ == '__main__':
    # global parameters
    ridge_para = 0.1
    dim_s = np.array([2,2,2,2])
    dim_s_history = np.cumsum(dim_s)
    dim_h = dim_s_history + np.arange(4)
    d_phi_1, d_phi_2, d_phi_3 = 2*(1+dim_s[0]), 2*(1+dim_s[1]), 2*(1+dim_s[2])
    k_max = 3

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
    
    W_stars = {1: W1_star, 2: W2_star, 3: W3_star}

    param_p_index = sys.argv.index('--p')
    param_n_index = sys.argv.index('--n')
    param_c_index = sys.argv.index('--c')
    p = float(sys.argv[param_p_index + 1])
    n = int(sys.argv[param_n_index + 1])
    c = float(sys.argv[param_c_index + 1])

    p_list = np.array([0.95, 0.75, 0.55])
    n_list = np.array([50,200,1000,5000,20000])
    c_list = np.array([0,5,10,50,100])

    p_index = np.argmax(p_list==p)
    n_index = np.argmax(n_list==n)
    c_index = np.argmax(c_list==c)
    job_index = p_index*25 + n_index*5 + c_index
    
    dim_s = np.array([2,2,2,2])
    dim_s_history = np.cumsum(dim_s)
    dim_h = dim_s_history + np.arange(4)
    
    
    N = 100
    T= 20
    print(f'Starting job {job_index} with parameters: p={p}, n={n}, c={c}')

    V_function_list = np.zeros((N, T+1))
    Q_learning_list = np.zeros(N)
    time_1 = time.time()

    random_seeds = np.random.choice(range(10000), size=N, replace=False)

    for i_seed in tqdm(range(N), desc="Processing seeds"):
        current_seed = random_seeds[i_seed]
        np.random.seed(current_seed)
        
        iter_start = time.time()
        D_offline = generate_offline_data(n, p, k_max, W_stars, dim_h, dim_s) 
        W1_hat, W2_hat, W3_hat = estimate_W(1, D_offline, ridge_para, dim_s, dim_h), estimate_W(2, D_offline, ridge_para, dim_s, dim_h), estimate_W(3, D_offline, ridge_para, dim_s, dim_h)
        W_hat_list = [W1_hat, W2_hat, W3_hat]
        
        theta_1 = np.zeros((L*dim_s_history[0], 2))
        theta_2 = np.zeros((L*dim_s_history[1], 4))
        theta_3 = np.zeros((L*dim_s_history[2], 8))
        theta_list = [theta_1, theta_2, theta_3]
        
        V_function_list[i_seed, 0] = compute_V1_unif(W_stars, theta_list, 150000, k_max, dim_s, dim_h, knots, L, k_degree)

        for t in tqdm(range(T), desc=f"Iterations for seed {i_seed}", leave=False):
                theta_list = one_iteration_parallel(
                    dim_s=dim_s,
                    dim_h=dim_h,
                    k_degree=k_degree,
                    knots=knots,
                    L=L,
                    theta_list=theta_list,
                    W_hat_list=W_hat_list,
                    eta=1,
                    n_MC=100,
                    m=500,
                    r_tilde=r_tilde,
                    k_max=3,
                    D_offline=D_offline,
                    ridge_para=0.1,
                    c_pessimism=5
                )
                V_function = compute_V1_unif(W_stars, theta_list, 150000, k_max, dim_s, dim_h, knots, L, k_degree)
                V_function_list[i_seed, t+1] = V_function
                
        theta_qlearn = standard_Q_learning(D_offline, k_max, dim_h, L, dim_s_history, knots, k_degree)
        Q_learning_list[i_seed] = compute_V1_unif(W_stars, theta_qlearn, 200000, k_max, dim_s, dim_h, knots, L, k_degree, greedy=True)
            
        iter_time = time.time() - iter_start
        if (i_seed + 1) % 10 == 0:  
            print(f'Completed {i_seed + 1}/{N} seeds. Average time per seed: {iter_time:.2f}s')

time_2 = time.time()
time_of_job = (time_2-time_1)/60

results_to_be_saved = np.concatenate((np.array([job_index]), np.array([time_of_job]), Q_learning_list,
                                    V_function_list.reshape(N*(T+1),)), axis=0)
pd.DataFrame(results_to_be_saved).to_csv('0112/result_'+str(job_index).zfill(2)+'.csv')

print('Job number {} is finished. Time: {:.3f} minutes'.format(job_index, time_of_job))
print(f'Results saved to: 0112/result_{str(job_index).zfill(2)}.csv')

