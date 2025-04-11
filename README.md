# POLAR 
This is a Python implementation of *POLAR: A Pessimistic Model-based Policy Learning Algorithm for Dynamic Treatment Regimes*

Authors: Ruijia Zhang, Zhengling Qi, Yue Wu, Xiangyu Zhang, Yanxun Xu

## Abstract: 
Dynamic treatment regimes (DTRs) provide a principled framework for optimizing sequential decision-making in domains where decisions must adapt over time in response to individual trajectories, such as healthcare, education, and digital interventions. However, existing statistical methods often rely on strong positivity assumptions and lack robustness under partial data coverage, while offline reinforcement learning approaches typically focus on average training performance, lack statistical guarantees, and require solving complex optimization problems. To address these challenges, we propose POLAR, a novel pessimistic model-based policy learning algorithm for offline DTR optimization. POLAR estimates the transition dynamics from offline data using parametric or nonparametric methods and quantifies uncertainty for each history-action pair. A pessimistic penalty is then incorporated into the reward function to discourage actions associated with high uncertainty. Unlike many existing methods that focus on average training performance, POLAR directly targets the suboptimality of the final learned policy and offers theoretical guarantees, without relying on computationally intensive minimax or constrained optimization procedures. To the best of our knowledge, POLAR is the first model-based DTR method to provide both statistical and computational guarantees, including finite-sample bounds on policy suboptimality. Empirical results on both synthetic data and the MIMIC-III dataset demonstrate that POLAR outperforms existing methods in terms of policy value and produces reliable, history-aware treatment strategies.

# Getting Started
Install the dependencies using:

```
pip install -r requirements.txt
```

# Run Instructions
To run the POLAR simulation, execute the following shell script:

```
bash simulation/submit_simulation.sh
```
This script will initiate the full pipeline for simulating environments, estimating transition models, and training the POLAR policy.





