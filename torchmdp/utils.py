import torch
from torch import nn



class Dimensions:
    def __init__(
        self, 
        num_observations=None, 
        num_observation_modalities=0, 
        num_states=None, 
        num_state_factors=0, 
        num_controls=None, 
        num_control_factors=0
    ):
        self.num_observations=num_observations
        self.num_observation_modalities=num_observation_modalities
        self.num_states=num_states
        self.num_state_factors=num_state_factors
        self.num_controls=num_controls
        self.num_control_factors=num_control_factors


def sample(probabilities):
    probabilities = probabilities.squeeze() if probabilities.dim() > 1 else probabilities
    sample_onehot = torch.multinomial(probabilities, 1)
    return torch.where(sample_onehot == 1)[0, 0]


def sample_obj_tensor(tens):
    samples = [sample(n) for n in tens]
    return samples


def random_A_matrix(num_obs, num_states, A_factor_list=None):
    if isinstance(num_obs, int):
        num_obs = [num_obs]
    if isinstance(num_states, int):
        num_states = [num_states]
    num_modalities = len(num_obs)

    if A_factor_list is None:
        num_factors = len(num_states)
        A_factor_list = [list(range(num_factors))] * num_modalities

    A = torch.empty(num_modalities)
    for modality, modality_obs in enumerate(num_obs):
        logging_dimensions = [num_states[idx] for idx in A_factor_list[modality]]
        modality_shape = [modality_obs] + logging_dimensions
        modality_dist = torch.rand(*modality_shape)
        A[modality] = norm_dist(modality_dist)