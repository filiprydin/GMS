import torch

def get_random_problems(distribution, batch_size, problem_size, emax):

    if problem_size == 21:
        demand_scaler = 30
    elif problem_size == 51:
        demand_scaler = 40
    elif problem_size == 101:
        demand_scaler = 50
    else:
        demand_scaler = 10

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    node_demand[:, 0] = 0 # Assume node 0 is depot
    # shape: (batch, problem)

    if distribution == "FIX":
        edge_attr, edge_indices = get_random_problems_fix(batch_size, problem_size, emax)
    if distribution == "FLEX":
        edge_attr, edge_indices = get_random_problems_flex(batch_size, problem_size, emax)
    elif distribution == "XASY":
        edge_attr, edge_indices = get_random_problems_XASY(batch_size, problem_size)

    # Return: 
    # (B, E, 2) edge distances/travel times
    # (B, E, 2) edge indices, to identify which edge goes where
    # (B, N) node demand
    return edge_attr, edge_indices, node_demand

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

def augment_data(dists, edge_to_node, node_demand, augmentation_factor = 8):
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
    aug_demands = node_demand
    for factor in possible_factors[1:]:
        aug_dists_new_1 = dists[:, :, 0]
        aug_dists_new_2 = dists[:, :, 1] * factor
        aug_dists_new = torch.stack((aug_dists_new_1, aug_dists_new_2), dim=-1)
        aug_dists = torch.cat((aug_dists, aug_dists_new), dim=0)

        aug_edge_to_node = torch.cat((aug_edge_to_node, edge_to_node), dim=0)
        aug_demands = torch.cat((aug_demands, node_demand), dim=0)

    return aug_dists, aug_edge_to_node, aug_demands, possible_factors