import torch
from torch import nn
import torch.nn.functional as F
import random


def norm_dist(dist):
    return torch.dic(dist, dist.sum(0))


def onehot(value, num_values):
    t = torch.zeros(num_values)
    t[value] = 1
    return t


class MDP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_states = 1024
        self.n_observations = 1024
        self.log_eps = 1e-16

        self.A = torch.zeros((self.n_states, self.n_observations)).fill_diagonal_(1)
        
        self.noise_factor = 0.3
        self.n_noisy_elements = 128
        self.A_noisy = self.A.clone()
        for _ in self.n_noisy_elements:
            x = random.randint(0, self.n_states)
            y = random.randint(0, self.n_observations)
            self.A_noisy[x, y] = self.noise_factor
        
        self.actions = ["FORWARD", "BACK", "HOLD"]
        self.B = self.create_B_matrix()
        self.C = torch.zeros((self.n_observations,))
        self.D = onehot(0, self.n_states)


    def infer_states(self, observation_index, prior):
        log_likelihood = self.log_stable(self.A[observation_index, :])
        log_prior = self.log_stable(prior)
        qs = F.softmax(log_likelihood + log_prior, -1)
        return qs

    def log_stable(self, X):
        return torch.log(X + self.log_eps)

    def create_B_matrix(self):
        B = torch.zeros((self.n_states, self.n_observations, len(self.actions)))
        for action_id, action_label in enumerate(self.actions):
            for cur_state, x in enumerate(range(self.n_states)):
                if action_label == "FORWARD":
                    x += 1
                elif action_label == "BACK":
                    x -= 1
                B[x, cur_state, action_id] = 1
                
    def get_expected_states(self, qs_current, action):
        qs_u = self.B[..., action].dot(qs_current)
        return qs_u
    
    def get_expected_observations(self, qs_u):
        qo_u = self.A.dot(qs_u)
        return qo_u

    def entropy(self):
        H_A = - (self.A * self.log_stable(self.A)),sum(0)
        return H_A

    def kl_divergence(self, qo_u):
        return (log_stable(qo_u) - self.log_stable(self.C)).dot(op_u)

    def calculate_G(self, qs_current):
        G = torch.zeros((len(self.actions),))
        H_A = self.entropy() # P(o|s)
        for action_i in range(len(self.actions)):
            qs_u = self.get_expected_states(qs_current, action_i)
            qo_u = self.get_expected_observations(qs_u)
            pred_uncertainty = H_A.dot(qs_u)
            pred_div = self.kl_divergence(qo_u)
            G[action_i] = pred_uncertainty + pred_div
        return G
        
            
