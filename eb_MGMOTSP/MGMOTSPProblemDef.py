import torch

def get_random_problems(distribution, batch_size, problem_size, emax):

    if distribution == "FIX":
        return get_random_problems_fix(batch_size, problem_size, emax)
    if distribution == "FLEX":
        return get_random_problems_flex(batch_size, problem_size, emax)

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

def augment_data(dists, edge_to_node, augmentation_factor = 8):
    if augmentation_factor > 1:
        step_size = 0.5 / (augmentation_factor // 2)

        possible_factors = [1]
        possible_factors.extend(
            [0.5 + x * step_size for x in range(augmentation_factor // 2)]
        )
        possible_factors.extend(
            [1.5 - x * step_size for x in range(augmentation_factor // 2)]
        )  ## 0.5 ... 1 ... 1.5
        
        possible_factors = possible_factors[:-1] # Exclude last so that aug factor matches specification

    aug_dists = dists
    aug_edge_to_node = edge_to_node
    for factor in possible_factors[1:]:
        aug_dists = torch.cat((aug_dists, dists * factor), dim=0)
        aug_edge_to_node = torch.cat((aug_edge_to_node, edge_to_node), dim=0)

    return aug_dists, aug_edge_to_node, possible_factors
