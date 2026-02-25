import torch

def get_random_problems(distribution, batch_size, problem_size):

    if distribution == "EUC":
        problems = torch.rand(size=(batch_size, problem_size, 4))
        return _get_problem_dists(problems)
    if distribution == "TMAT":  
        matrix1 = _generate_TMAT(batch_size, problem_size)
        matrix2 = _generate_TMAT(batch_size, problem_size)
        problems = torch.stack((matrix1, matrix2), dim=3)
        return problems
    if distribution == "XASY":
        problems = torch.rand((batch_size, problem_size, problem_size, 2))
        return problems

def _generate_TMAT(batch_size, problem_size, min_val=1, max_val=1000000):
    problems = torch.randint(low=min_val, high=max_val+1, size=(batch_size,
        problem_size, problem_size))
    problems[:, torch.arange(problem_size), torch.arange(problem_size)] = 0
    while True:
        old_problems = problems.clone()
        problems, _ = (problems[:, :, None, :] + problems[:, None, :,
            :].transpose(2,3)).min(dim=3)
        if (problems == old_problems).all():
            break

    max_value = problems.amax(dim=(1, 2), keepdim=True)  # Shape (B, 1, 1)
    problems_max = max_value.expand(batch_size, problem_size, problem_size)

    return torch.divide(problems, problems_max)

def _get_problem_dists(problems):
    B, N, Nobj2 = problems.shape
    
    x_coords = problems[:, :, ::2]  # shape (B, N, Nobj)
    y_coords = problems[:, :, 1::2]  # shape (B, N, Nobj)


    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    
    x_diff = x_coords[:, idx_i] - x_coords[:, idx_j]  # shape (B, num_edges, Nobj)
    y_diff = y_coords[:, idx_i] - y_coords[:, idx_j]  # shape (B, num_edges, Nobj)

    x_diff = x_coords.unsqueeze(2) - x_coords.unsqueeze(1)  # Shape: (B, N, N, 2)
    y_diff = y_coords.unsqueeze(2) - y_coords.unsqueeze(1)  # Shape: (B, N, N, 2)
    dist_matrix = torch.sqrt(x_diff ** 2 + y_diff ** 2)  # Shape: (B, N, N, 2)

    return dist_matrix

def augment_data(dist_matrix, augmentation_factor = 8):
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

    aug_dist_matrix = dist_matrix
    for factor in possible_factors[1:]:
        aug_dist_matrix = torch.cat((aug_dist_matrix, dist_matrix * factor), dim=0)

    return aug_dist_matrix, possible_factors