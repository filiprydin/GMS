import torch

def get_random_problems(distribution, batch_size, problem_size):

    if distribution == "EUC":
        problems = torch.rand(size=(batch_size, problem_size, 2))
        matrix = _get_problem_dists(problems).squeeze(3)
    elif distribution == "TMAT":  
        matrix = _generate_TMAT(batch_size, problem_size)
    elif distribution == "XASY":
        matrix = torch.rand((batch_size, problem_size, problem_size))
        matrix[:, torch.arange(problem_size), torch.arange(problem_size)] = 0
    # shape: (batch, problem, problem)

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

    return matrix, node_demand

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
    B, N, _ = problems.shape
    
    x_coords = problems[:, :, ::2]  # shape (B, N, Nobj)
    y_coords = problems[:, :, 1::2]  # shape (B, N, Nobj)


    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    
    x_diff = x_coords[:, idx_i] - x_coords[:, idx_j]  # shape (B, num_edges)
    y_diff = y_coords[:, idx_i] - y_coords[:, idx_j]  # shape (B, num_edges)

    x_diff = x_coords.unsqueeze(2) - x_coords.unsqueeze(1)  # Shape: (B, N, N)
    y_diff = y_coords.unsqueeze(2) - y_coords.unsqueeze(1)  # Shape: (B, N, N)
    dist_matrix = torch.sqrt(x_diff ** 2 + y_diff ** 2)  # Shape: (B, N, N)

    return dist_matrix

def augment_data(dist_matrix, demands, augmentation_factor = 8):
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
    aug_demands = demands
    for factor in possible_factors[1:]:
        aug_dist_matrix = torch.cat((aug_dist_matrix, dist_matrix * factor), dim=0)
        aug_demands = torch.cat((aug_demands, demands), dim=0)

    return aug_dist_matrix, aug_demands, possible_factors