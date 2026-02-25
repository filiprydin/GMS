import torch
import numpy as np
from collections import namedtuple

def get_random_problems(distribution, batch_size, problem_size, emax):

    if distribution == "FIX":
        edge_attr, edge_indices = get_random_problems_fix(batch_size, problem_size, emax)
    elif distribution == "FLEX":
        edge_attr, edge_indices = get_random_problems_flex(batch_size, problem_size, emax)
    elif distribution == "XASY":
        edge_attr, edge_indices = get_random_problems_XASY(batch_size, problem_size)
        emax = 1

    # We use medium instances below to sample time windows
    # Ignore coordinates
    # The time factor depends on how we sample distances. 
    # According to Chen et. al. (2024) it should correspond to expected tour duration of a random tsp tour
    # Service time is always 0

    # Estimated time factors: 
    if distribution == "FLEX":
        if problem_size == 21:
            if emax == 2:
                time_factor = 8.3
            elif emax == 5:
                time_factor = 6.7
            elif emax == 10:
                time_factor = 5.9
        elif problem_size == 51:
            if emax == 2:
                time_factor = 20.8
            elif emax == 5:
                time_factor = 16.6
            elif emax == 10:
                time_factor = 14.8
        elif problem_size == 101:
            if emax == 2:
                time_factor = 41.7
            elif emax == 5:
                time_factor = 33.4
            elif emax == 10:
                time_factor = 29.5
        else:
            time_factor = 10
    elif distribution == "FIX":
        if problem_size == 21:
            time_factor = 10
        elif problem_size == 51:
            time_factor = 25
        elif problem_size == 101:
            time_factor = 50
        else:
            time_factor = 10
    elif distribution == "XASY":
        if problem_size == 21:
            time_factor = 10.0
        elif problem_size == 51:
            time_factor = 25
        elif problem_size == 101:
            time_factor = 50
        else:
            time_factor = 10
    else:
        time_factor = 10
    
    hardness = "medium"
    loc_factor = 1000
    tw = generate_tsptw_data(size=batch_size, graph_size=problem_size, time_factor=time_factor*loc_factor, loc_factor=loc_factor, tw_duration="5075" if hardness == "easy" else "1020")
    
    loc_factor = tw.loc_factor[0] 
    time_windows = torch.tensor(tw.node_tw) / loc_factor
    service_time = torch.zeros(size=(batch_size,problem_size))

    tw_start, tw_end = time_windows[:,:,0], time_windows[:,:,1]

    # Upper bound for depot = max(node ub + 1), to make this tight´
    tw_end[:, 0] = (1 + tw_end[:, 1:]).max(dim=-1)[0]

    # Return: 
    # (B, E, 2) edge distances/travel times
    # (B, E, 2) edge indices, to identify which edge goes where
    # (B, N) service times
    # (B, N) time window start times
    # (B, N) time window end times
    return edge_attr, edge_indices, service_time, tw_start, tw_end


### Time-windows ###
####################

def gen_tw(size, graph_size, time_factor, dura_region, rnds):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """

    service_window = int(time_factor * 2)

    horizon = np.zeros((size, graph_size, 2))
    horizon[:] = [0, service_window]

    # sample earliest start times
    tw_start = rnds.randint(horizon[..., 0], horizon[..., 1] / 2)
    tw_start[:, 0] = 0

    # calculate latest start times b, which is
    # tw_start + service_time_expansion x normal random noise, all limited by the horizon
    # and combine it with tw_start to create the time windows
    epsilon = rnds.uniform(dura_region[0], dura_region[1], (tw_start.shape))
    duration = np.around(time_factor * epsilon)
    duration[:, 0] = service_window
    tw_end = np.minimum(tw_start + duration, horizon[..., 1]).astype(int)

    tw = np.concatenate([tw_start[..., None], tw_end[..., None]], axis=2).reshape(size, graph_size, 2)

    return tw

def generate_tsptw_data(size, graph_size, rnds=None, time_factor=100.0, loc_factor=100, tw_duration="5075"):
    """
    Copyright (c) 2020 Jonas K. Falkner
    Copy from https://github.com/jokofa/JAMPR/blob/master/data_generator.py
    """

    rnds = np.random if rnds is None else rnds
    service_window = int(time_factor * 2)

    # sample locations
    nloc = rnds.uniform(size=(size, graph_size, 2)) * loc_factor  # node locations

    # tw duration
    dura_region = {
         "5075": [.5, .75],
         "1020": [.1, .2],
    }
    if isinstance(tw_duration, str):
        dura_region = dura_region[tw_duration]
    else:
        dura_region = tw_duration

    tw = gen_tw(size, graph_size, time_factor, dura_region, rnds)

    return TSPTW_SET(node_loc=nloc,
                     node_tw=tw,
                     durations=tw[..., 1] - tw[..., 0],
                     service_window=[service_window] * size,
                     time_factor=[time_factor] * size,
                     loc_factor=[loc_factor] * size, )

TSPTW_SET = namedtuple("TSPTW_SET",
                       ["node_loc",  # Node locations 1
                        "node_tw",  # node time windows 5
                        "durations",  # service duration per node 6
                        "service_window",  # maximum of time units 7
                        "time_factor", "loc_factor"])


def get_random_problems_fix(batch_size, problem_size, emax):

    matrix = torch.rand(batch_size, problem_size * (problem_size - 1), emax, 2)
    problems, _ = torch.sort(matrix, dim=2)  # Sort along each dimension
    problems[:, :, :, 1] = problems[:, :, :, 1].flip(dims=(2,))

    idx_i, idx_j = torch.triu_indices(problem_size, problem_size, offset=1)
    idx_pairs = torch.stack([idx_i, idx_j], dim=1)  # shape (num_edges, 2)
    reverse_idx_pairs = torch.stack([idx_j, idx_i], dim=1)  # shape (num_edges, 2)
    all_idx_pairs = torch.cat([idx_pairs, reverse_idx_pairs], dim=1).view(-1, 2)  # Flatten into pairs
    indices = all_idx_pairs.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.int32)

    edge_dists = problems.reshape(batch_size, emax * (problem_size * (problem_size - 1)), 2)
    edge_indices = indices.repeat_interleave(emax, dim=1)

    return edge_dists, edge_indices

def get_random_problems_flex(batch_size, problem_size, emax):
    matrix = torch.rand(batch_size, problem_size * (problem_size - 1), emax, 2)
    result = matrix.clone()

    # Filter out all dominated edges
    for i in range(emax): # For each parallel edge
        for j in range(emax): # Loop through all other edges
            if i != j:
                # Replace dominated edges
                slice_i = matrix[..., i, :]  # Shape: (B, E, 2)
                slice_j = matrix[..., j, :]  # Shape: (B, E, 2)

                condition_j_greater = (slice_j > slice_i).all(dim=-1)  # Shape: (B, E)

                indices_j_greater = condition_j_greater.nonzero(as_tuple=True)

                result[indices_j_greater + (torch.tensor(j), slice(None))] = matrix[indices_j_greater + (torch.tensor(i), slice(None))]

    idx_i, idx_j = torch.triu_indices(problem_size, problem_size, offset=1)
    idx_pairs = torch.stack([idx_i, idx_j], dim=1)  # shape (num_edges, 2)
    reverse_idx_pairs = torch.stack([idx_j, idx_i], dim=1)  # shape (num_edges, 2)
    all_idx_pairs = torch.cat([idx_pairs, reverse_idx_pairs], dim=1).view(-1, 2)  # Flatten into pairs
    indices = all_idx_pairs.unsqueeze(0).repeat(batch_size, 1, 1).to(torch.int32)

    edge_dists = result.reshape(batch_size, emax * (problem_size * (problem_size - 1)), 2)
    edge_indices = indices.repeat_interleave(emax, dim=1)

    return edge_dists, edge_indices

def get_random_problems_XASY(batch_size, problem_size):
    problems = torch.rand(batch_size, problem_size, problem_size, 2)

    return _get_problem_dists(problems)

def _get_problem_dists(problems):
    """ 
    Converts set of distance matrices (B, N, N, Nobj) to edge attributes (B, E, Nobj) and edge indices (B, E, 2)
    """
    B, N, _, Nobj = problems.shape

    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    idx_pairs = torch.stack([idx_i, idx_j], dim=1)  # shape (num_edges, 2)
    reverse_idx_pairs = torch.stack([idx_j, idx_i], dim=1)  # shape (num_edges, 2)
    all_idx_pairs = torch.cat([idx_pairs, reverse_idx_pairs], dim=1).view(-1, 2)  # Flatten into pairs
    indices = all_idx_pairs.unsqueeze(0).repeat(B, 1, 1).to(torch.int32)
    i_indices, j_indices = all_idx_pairs[:, 0], all_idx_pairs[:, 1]

    dists = problems[:, i_indices, j_indices, :]

    return dists, indices

def augment_data(dists, edge_to_node, service_time, tw_start, tw_end, augmentation_factor = 8):
    if augmentation_factor > 1:
        step_size = 0.5 / (augmentation_factor // 2)

        possible_factors = [1]
        possible_factors.extend(
            [0.5 + x * step_size for x in range(augmentation_factor // 2)]
        )
        possible_factors.extend(
            [1.5 - x * step_size for x in range(augmentation_factor // 2)]
        )  ## 0.5 ... 1 ... 1.5
        
        #factor = random.choice(possible_factors)
        possible_factors = possible_factors[:-1] # Exclude last so that aug factor matches specification

    aug_dists = dists
    aug_edge_to_node = edge_to_node
    aug_service_time = service_time
    aug_tw_start = tw_start
    aug_tw_end = tw_end
    for factor in possible_factors[1:]:
        aug_dists_new_1 = dists[:, :, 0]
        aug_dists_new_2 = dists[:, :, 1] * factor
        aug_dists_new = torch.stack((aug_dists_new_1, aug_dists_new_2), dim=-1)
        aug_dists = torch.cat((aug_dists, aug_dists_new), dim=0)

        aug_edge_to_node = torch.cat((aug_edge_to_node, edge_to_node), dim=0)
        aug_service_time = torch.cat((aug_service_time, service_time), dim=0)
        aug_tw_start = torch.cat((aug_tw_start, tw_start), dim=0)   
        aug_tw_end = torch.cat((aug_tw_end, tw_end), dim=0)

    return aug_dists, aug_edge_to_node, aug_service_time, aug_tw_start, aug_tw_end, possible_factors