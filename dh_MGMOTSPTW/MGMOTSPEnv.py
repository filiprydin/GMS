from dataclasses import dataclass
import torch

from MGMOTSPProblemDef import get_random_problems


@dataclass
class Reset_State:
    dist_matrix: torch.Tensor
    # shape: (batch, N, N, 2)
    selected_edges: torch.Tensor
    # shape: (batch, N*(N-1))

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    selected_count: torch.Tensor = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, N)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.distribution = env_params['distribution']
        self.emax_max = env_params['emax']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.edge_attr = None # shape: (batch, E, 2)
        self.edge_to_node = None # shape: (batch, E, 2)
        self.service_time = None # shape: (batch, N)
        self.tw_start = None # shape: (batch, N)
        self.tw_end = None # shape: (batch, N)

        # Const @Sparsification
        self.dist_matrix = None

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        self.timestamps = None
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.current_time = None # shape: (batch, P)

    def load_problems(self, batch_size, simple_graph = False, simple_graph_dist = None, aug_factor=1):
        self.batch_size = batch_size

        if not simple_graph:
            self.edge_attr, self.edge_to_node, self.service_time, self.tw_start, self.tw_end = get_random_problems(self.distribution, batch_size, self.problem_size, self.emax_max)
            self.emax = self.emax_max
        else: 
            self.edge_attr, self.edge_to_node, self.service_time, self.tw_start, self.tw_end = get_random_problems(simple_graph_dist, batch_size, self.problem_size, 1)
            self.emax = 1

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self, selected_edges):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.timestamps = torch.zeros((self.batch_size, self.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size))

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, E)
        self.step_state.selected_count = 0

        reward = None
        done = False 

        # Get distances and convert to matrix
        n_edges = self.problem_size * (self.problem_size - 1)
        selected_dists = torch.gather(self.edge_attr, 1, selected_edges.unsqueeze(2).expand(self.batch_size, n_edges, 2)) # (B, N*(N-1), 2)
        edge_to_node = torch.gather(self.edge_to_node, 1, selected_edges.unsqueeze(2).expand(self.batch_size, n_edges, 2))
        idx_from = edge_to_node[:, :, 0]
        idx_to = edge_to_node[:, :, 1]
        self.dist_matrix = torch.zeros(self.batch_size, self.problem_size, self.problem_size, 2)
        batch_idx = torch.arange(self.batch_size)[:, None].expand(self.batch_size, n_edges)
        self.dist_matrix[batch_idx, idx_from, idx_to, :] = selected_dists

        reset_state = Reset_State(self.dist_matrix, selected_edges)

        return reset_state, reward, done

    def pre_step(self):
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        if self.selected_count > 0:
            old_node = self.current_node.clone()

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, E)
        if self.selected_count == 1: # Special case for first selected, which is always the depot
            # Current time initialized as zeros -> pass
            pass
        else:
            new_time = self.dist_matrix[self.BATCH_IDX, old_node, self.current_node, 0] # (B, P) NOTE: We assume cost 0 is the time duration 
            tw_start = self.tw_start[self.BATCH_IDX, self.current_node] # (B, P)
            service_time = self.service_time[self.BATCH_IDX, self.current_node] # (B, P)
            #self.current_time = torch.maximum(self.current_time + new_time, tw_start) + service_time
            self.current_time = self.current_time + new_time + service_time
            self.step_state.current_time = self.current_time

        self.timestamps = torch.cat((self.timestamps, self.current_time[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)
        # NOTE: we include service time in violation calculation -> entire service must be within time window

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward_dist = -self._get_travel_distance()  # note the minus sign!
            reward_tw = -self._get_tw_violation()
            reward = torch.stack((reward_tw, reward_dist), dim=2)
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        # tours: shape (B, P, N)
        indices_from = self.selected_node_list[:, :, :-1]  # (B, P, N-1) - starting nodes of transitions
        indices_to = self.selected_node_list[:, :, 1:]     # (B, P, N-1) - ending nodes of transitions
        indices_from_last = self.selected_node_list[:, :, -1]  # (B, P) - last node of each tour
        indices_to_first = self.selected_node_list[:, :, 0]     # (B, P) - first node of each tour

        all_indices_from = torch.cat((indices_from, indices_from_last.unsqueeze(2)), dim=2)  # (B, P, N)
        all_indices_to = torch.cat((indices_to, indices_to_first.unsqueeze(2)), dim=2)  # (B, P, N)

        # Shape (B, P, N)
        from_distances = self.dist_matrix[torch.arange(self.batch_size).unsqueeze(1).unsqueeze(2), all_indices_from, all_indices_to, 1]
        total_distances = from_distances.sum(dim=2)

        # Shape (B, P)
        return total_distances
    
    def _get_tw_violation(self):

        # Ordered depending on visitation order
        tw_end_selected = self.tw_end[self.BATCH_IDX.unsqueeze(2), self.selected_node_list] # (B, P, N)
        tw_start_selected = self.tw_start[self.BATCH_IDX.unsqueeze(2), self.selected_node_list] # (B, P, N)

        violation_late = (self.timestamps - tw_end_selected > 0).float()
        violation_early = (self.timestamps - tw_start_selected < 0).float()

        total_violation = violation_late.sum(dim=2) + violation_early.sum(dim=2)

        # Add violation of final link back to depot
        final_nodes = self.selected_node_list[:, :, -1]
        depot = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.int64)
        new_time = self.dist_matrix[self.BATCH_IDX, final_nodes, depot, 0] # (B, P)
        tw_start = self.tw_start[self.BATCH_IDX, depot] # (B, P)
        tw_end = self.tw_end[self.BATCH_IDX, depot] # (B, P)
        service_time = self.service_time[self.BATCH_IDX, depot] # (B, P)
        final_time = self.current_time + new_time + service_time
        final_violation_late = (final_time - tw_end > 0).float()
        final_violation_early = (final_time - tw_start < 0).float()

        # Shape (B, P)
        return total_violation + final_violation_late + final_violation_early