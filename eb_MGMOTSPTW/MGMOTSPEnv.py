from dataclasses import dataclass
import torch

from MGMOTSPProblemDef import get_random_problems


@dataclass
class Reset_State:
    edge_attr: torch.Tensor
    # shape: (batch, E, 2)
    service_times: torch.Tensor
    tw_start: torch.Tensor
    tw_end: torch.Tensor
    # shape: (batch, problem)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    selected_count: torch.Tensor = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, E)
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
        self.emax = env_params['emax']
        self.edge_size = self.emax * self.problem_size * (self.problem_size - 1)

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

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_edge_list = None
        # shape: (batch, pomo, 0~problem)
        self.timestamps = None
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.current_time = None # shape: (batch, P)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.edge_attr, self.edge_to_node, self.service_time, self.tw_start, self.tw_end = get_random_problems(self.distribution, batch_size, self.problem_size, self.emax)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_edge_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.timestamps = torch.zeros((self.batch_size, self.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size))

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.edge_size))
        # shape: (batch, pomo, E)
        self.step_state.selected_count = 0

        # Choose first node -> depot
        # Mask out until last step
        self.starting_node = torch.zeros((self.batch_size, self.pomo_size), dtype = torch.int64)
        self.current_node = self.starting_node
        self.step_state.current_node = self.current_node

        edges_to_mask = find_mask_indices(self.edge_to_node, self.current_node) # (B, P, emax*(N-1))
        batch_idx_expanded = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        pomo_idx_expanded = self.POMO_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        self.step_state.ninf_mask[batch_idx_expanded, pomo_idx_expanded, edges_to_mask] = float('-inf')

        reward = None
        done = False 

        reset_state = Reset_State(self.edge_attr, self.service_time, self.tw_start, self.tw_end)

        return reset_state, reward, done

    def pre_step(self):
        self.step_state.current_node = self.current_node
        self.step_state.current_time = self.current_time

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        edge_to_node_expanded = self.edge_to_node[:, :, 1].unsqueeze(1).expand(self.batch_size, self.pomo_size, self.edge_size) # (B, P, E)
        self.current_node = torch.gather(edge_to_node_expanded, 2, selected.unsqueeze(-1))[:, :, 0]
        # shape: (batch, pomo)
        self.selected_edge_list = torch.cat((self.selected_edge_list, selected[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        edges_to_mask = find_mask_indices(self.edge_to_node, self.current_node) # (B, P, emax*(N-1))
        batch_idx_expanded = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        pomo_idx_expanded = self.POMO_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        self.step_state.ninf_mask[batch_idx_expanded, pomo_idx_expanded, edges_to_mask] = float('-inf')
        # shape: (batch, pomo, E)

        # If selecting last step, unmask edges to starting node
        if self.selected_count == self.problem_size - 1:
            edges_to_mask = find_mask_indices(self.edge_to_node, self.starting_node)
            self.step_state.ninf_mask[batch_idx_expanded, pomo_idx_expanded, edges_to_mask] = 0

        new_time = self.edge_attr[self.BATCH_IDX, selected, 0] # (B, P) NOTE: We assume cost 0 is the time duration 
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
        # tours: shape (B, P, N), edge indices
        problems_expanded = self.edge_attr.unsqueeze(1).expand(self.batch_size, self.pomo_size, self.edge_size, 2)
        selected_edge_list_expanded = self.selected_edge_list.unsqueeze(-1).expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        distances = torch.gather(problems_expanded, 2, selected_edge_list_expanded)[:, :, :, 1]
        # shape (B, P, N)

        # Shape (B, P)
        return torch.sum(distances, dim=2)
    
    def _get_tw_violation(self):

        edge_to_node_expanded = self.edge_to_node.unsqueeze(1).expand(self.batch_size, self.pomo_size, self.edge_size, 2)[:, :, :, 1]
        selected_node_list = torch.gather(edge_to_node_expanded, 2, self.selected_edge_list) # (B, P, N)
        # Ordered depending on visitation order
        tw_end_selected = self.tw_end[self.BATCH_IDX.unsqueeze(2), selected_node_list] # (B, P, N)
        tw_start_selected = self.tw_start[self.BATCH_IDX.unsqueeze(2), selected_node_list] # (B, P, N)

        # If timestamp < tw_end add 0, otherwise add violation
        #violation = torch.maximum(torch.zeros(self.batch_size, self.pomo_size, self.problem_size), self.timestamps - tw_end_selected) # (B, P, N)
        violation_late = (self.timestamps - tw_end_selected > 0).float()
        violation_early = (self.timestamps - tw_start_selected < 0).float()

        total_violation = violation_late.sum(dim=2) + violation_early.sum(dim=2)

        # NOTE: No final violation as last selected corresponds to an edge back to depot -> last step handled above

        # Shape (B, P)
        return total_violation
    
### Utils ###

def find_mask_indices(edge_to_node, selected_nodes):
    B, E, _ = edge_to_node.shape
    _, P = selected_nodes.shape

    # Condition: Must go to the current node
    matches = (selected_nodes.unsqueeze(-1) == edge_to_node[:,:,1].unsqueeze(1))  # Shape (B, P, E)
    index_array = torch.arange(E, device=edge_to_node.device).expand(B, P, E)
    edges_to_mask = index_array[matches].reshape(B, P, -1)

    return edges_to_mask