from dataclasses import dataclass
import torch

from MGMOTSPProblemDef import get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, E, 2)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, E)


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
        self.problems = None # shape: (batch, E, 2)
        self.edge_to_node = None # shape: (batch, E, 2)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_edge_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.problems, self.edge_to_node = get_random_problems(self.distribution, batch_size, self.problem_size, self.emax) # Shape (B, E, 2)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_edge_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.edge_size))
        # shape: (batch, pomo, E)

        # Choose first node -> POMO
        # Mask out until last step
        self.starting_node = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.current_node = self.starting_node
        self.step_state.current_node = self.current_node

        edges_to_mask = find_mask_indices(self.edge_to_node, self.current_node) # (B, P, emax*(N-1))
        batch_idx_expanded = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        pomo_idx_expanded = self.POMO_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        self.step_state.ninf_mask[batch_idx_expanded, pomo_idx_expanded, edges_to_mask] = float('-inf')

        reward = None
        done = False 

        reset_state = Reset_State(self.problems)

        return reset_state, reward, done

    def pre_step(self):
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

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        # tours: shape (B, P, N), edge indices
        problems_expanded = self.problems.unsqueeze(1).expand(self.batch_size, self.pomo_size, self.edge_size, 2)
        selected_edge_list_expanded = self.selected_edge_list.unsqueeze(-1).expand(self.batch_size, self.pomo_size, self.problem_size, 2)
        distances = torch.gather(problems_expanded, 2, selected_edge_list_expanded) 
        # shape (B, P, N, 2)

        # Shape (B, P, 2)
        return torch.sum(distances, dim=2)
    
### Utils ###

def find_mask_indices(edge_to_node, selected_nodes):
    B, E, _ = edge_to_node.shape
    _, P = selected_nodes.shape

    # Condition: Must go to the current node
    matches = (selected_nodes.unsqueeze(-1) == edge_to_node[:,:,1].unsqueeze(1))  # Shape (B, P, E)
    index_array = torch.arange(E, device=edge_to_node.device).expand(B, P, E)
    edges_to_mask = index_array[matches].reshape(B, P, -1)

    return edges_to_mask