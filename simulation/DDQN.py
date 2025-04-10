
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
from utils import *


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

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)


class DDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 memory_size=10000, batch_size=128, tau=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau  # Soft update coefficient
        self.memory = deque(maxlen=memory_size)

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.update_count = 0

    def select_action(self, state, training=True):
        # epsilon-greedy
        if training and random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def soft_update_target_network(self):
        """
        Soft update target network parameters:
        target_param = tau * local_param + (1 - tau) * target_param
        """
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions)

        target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Perform soft update on the target network
        self.soft_update_target_network()

        # Update epsilon for exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()
        
def prepare_memory(agent, s1, s2, s3, s4, a1, a2, a3, r1, r2, r3):

    # Store transitions
    for i in range(len(s1)):
        # First transition: (s1, a1, r1, s2, done=False)
        agent.store_transition(s1[i], a1[i], r1[i], s2[i], False)
        # Second transition: (s2, a2, r2, s3, done=False)
        agent.store_transition(s2[i], a2[i], r2[i], s3[i], False)
        # Third transition: (s3, a3, r3, s4, done=True)
        agent.store_transition(s3[i], a3[i], r3[i], s4[i], True)


def compute_validation_loss(agent, s1_val, s2_val, s3_val, s4_val, a1_val, a2_val, a3_val, r1_val, r2_val, r3_val):
    """
    Compute validation loss using validation data and rewards r1, r2, r3.
    """
    agent.q_network.eval()
    with torch.no_grad():
        # Build validation transitions
        val_transitions = []
        for i in range(len(s1_val)):
            # Transition 1
            val_transitions.append((s1_val[i], a1_val[i], r1_val[i], s2_val[i], False))
            # Transition 2
            val_transitions.append((s2_val[i], a2_val[i], r2_val[i], s3_val[i], False))
            # Transition 3
            val_transitions.append((s3_val[i], a3_val[i], r3_val[i], s4_val[i], True))

        # Pack validation transitions
        states, actions, rewards, next_states, dones = zip(*val_transitions)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute Q-values and targets
        current_q_values = agent.q_network(states).gather(1, actions.unsqueeze(1))
        next_actions = agent.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = agent.target_network(next_states).gather(1, next_actions)
        target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * agent.gamma * next_q_values

        # Calculate validation loss
        loss_fn = nn.MSELoss()
        val_loss = loss_fn(current_q_values, target_q_values)
    agent.q_network.train()
    return val_loss.item()




def offline_train_ddqn(agent, s1_train, s2_train, s3_train, s4_train, a1_train, a2_train, a3_train, h2_train, h3_train, r1_train, r2_train, r3_train,
                       s1_val, s2_val, s3_val, s4_val, a1_val, a2_val, a3_val, h2_val, h3_val, r1_val, r2_val, r3_val, h1, batch_size,
                       num_epochs=2000, patience=20):
    losses = []
    val_losses = []

    best_model_state = None
    best_val_loss = float("inf")
    no_improve_count = 0

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Step 1: Update the model
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        val_loss = compute_validation_loss(agent, s1_val, s2_val, s3_val, s4_val,
                                           a1_val, a2_val, a3_val, r1_val, r2_val, r3_val)
        val_losses.append(val_loss)

        # Step 3: Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = copy.deepcopy(agent.q_network.state_dict())  # Save best model
        else:
            no_improve_count += 1

        # Step 4: Early stopping condition
        if no_improve_count >= patience:
            print(f"Validation loss has not improved for {patience} epochs, stopping early.")
            break

    if best_model_state is not None:
        agent.q_network.load_state_dict(best_model_state)
        agent.q_network.eval()
        ope_value = evaluation(agent, h1, batch_size, greedy=True)
        print(f"Evaluation value (Best Model): {ope_value}")
    else:
        print("No best model found (check if training didn't run?), cannot compute OPE.")

    return losses, val_losses


def evaluation(agent, h1, batch_size, greedy=True):

    h1_tensor = torch.FloatTensor(h1)
    with torch.no_grad():
        q1 = agent.q_network(h1_tensor)
        a1 = q1.argmax(dim=1).cpu().numpy()

    r1 = r_true_expected(k=1, h=h1, a=a1)

    s2 = get_transition(k=1, W=W1_star, h=h1, a=a1)
    s2_tensor = torch.FloatTensor(s2)
    with torch.no_grad():
        q2 = agent.q_network(s2_tensor)
        a2 = q2.argmax(dim=1).cpu().numpy()

    h2 = np.concatenate([h1, a1.reshape(-1, 1), s2], axis=1)
    s3 = get_transition(k=2, W=W2_star, h=h2, a=a2)
    r2 = r_true_expected(k=2, h=h2, a=a2)

    s3_tensor = torch.FloatTensor(s3)
    with torch.no_grad():
        q3 = agent.q_network(s3_tensor)
        a3 = q3.argmax(dim=1).cpu().numpy()

    h3 = np.concatenate([h2, a2.reshape(-1, 1), s3], axis=1)
    r3 = r_true_expected(k=3, h=h3, a=a3)

    total_return = r1 + r2 + r3
    return np.mean(total_return)




def offline_train_ddqn(agent, s1_train, s2_train, s3_train, s4_train, a1_train, a2_train, a3_train, h2_train, h3_train, r1_train, r2_train, r3_train,
                       s1_val, s2_val, s3_val, s4_val, a1_val, a2_val, a3_val, h2_val, h3_val, r1_val, r2_val, r3_val, h1, batch_size,
                       num_epochs=2000, patience=20):
    losses = []
    val_losses = []

    best_model_state = None
    best_val_loss = float("inf")
    no_improve_count = 0

    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Step 1: Update the model
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        val_loss = compute_validation_loss(agent, s1_val, s2_val, s3_val, s4_val,
                                           a1_val, a2_val, a3_val, r1_val, r2_val, r3_val)
        val_losses.append(val_loss)

        # Step 3: Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = copy.deepcopy(agent.q_network.state_dict())  # Save best model
        else:
            no_improve_count += 1

        # Step 4: Early stopping condition
        if no_improve_count >= patience:
            print(f"Validation loss has not improved for {patience} epochs, stopping early.")
            break

    if best_model_state is not None:
        agent.q_network.load_state_dict(best_model_state)
        agent.q_network.eval()
        ope_value = evaluation(agent, h1, batch_size, greedy=True)
        print(f"Evaluation value (Best Model): {ope_value}")
    else:
        print("No best model found (check if training didn't run?), cannot compute OPE.")

    return losses, val_losses




