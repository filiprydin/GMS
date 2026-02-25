from dataclasses import dataclass
import torch

from eb_MGMOCVRP.MGMOCVRPProblemDef import get_random_problems


@dataclass
class Reset_State:
    edge_attr: torch.Tensor
    # shape: (batch, E, 2)
    demands: torch.Tensor
    # shape: (batch, N)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    selected_count: torch.Tensor = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, E)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
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
        self.demands = None # shape: (batch, N)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_edge_list = None
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag_nodes = None
        # shape: (batch, pomo, problem)
        self.ninf_mask_nodes = None
        # shape: (batch, pomo, problem)
        self.finished = None
        # shape: (batch, pomo)
        self.finished_step = None
        # shape: (batch, pomo)

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        self.edge_attr, self.edge_to_node, self.demands = get_random_problems(self.distribution, batch_size, self.problem_size, self.emax)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_edge_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag_nodes = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)
        self.ninf_mask_nodes = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.finished_step = None
        # shape: (batch, pomo)

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

        edges_to_mask = find_mask_indices_single(self.edge_to_node, self.current_node) # (B, P, emax*(N-1))
        batch_idx_expanded = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        pomo_idx_expanded = self.POMO_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.emax * (self.problem_size - 1))
        self.step_state.ninf_mask[batch_idx_expanded, pomo_idx_expanded, edges_to_mask] = float('-inf')

        reward = None
        done = False 

        reset_state = Reset_State(self.edge_attr, self.demands)

        return reset_state, reward, done

    def pre_step(self):
        self.step_state.current_node = self.current_node
        self.step_state.finished = self.finished
        self.step_state.load = self.load

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected_edges):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        edge_to_node_expanded = self.edge_to_node[:, :, 1].unsqueeze(1).expand(self.batch_size, self.pomo_size, self.edge_size) # (B, P, E)
        self.current_node = torch.gather(edge_to_node_expanded, 2, selected_edges.unsqueeze(-1))[:, :, 0]
        # shape: (batch, pomo)
        self.selected_edge_list = torch.cat((self.selected_edge_list, selected_edges[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

         # Dynamic-2
        ####################################
        self.at_the_depot = (self.current_node == 0)

        demand_list = self.demands[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem)
        gathering_index = self.current_node[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag_nodes[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, problem)
        self.visited_ninf_flag_nodes[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask_nodes = self.visited_ninf_flag_nodes.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem)
        self.ninf_mask_nodes[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem)

        newly_finished = (self.visited_ninf_flag_nodes == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # Register which step each episode finished
        if self.finished_step is None:
            self.finished_step = torch.zeros_like(self.finished, dtype=torch.long)
        else:
            self.finished_step[self.finished & self.finished_step.eq(0)] = self.selected_count

        # do not mask depot for finished episodes.
        self.ninf_mask_nodes[:, :, 0][self.finished] = 0

        # UPDATE STEP STATE
        #####################################
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.finished = self.finished
        self.step_state.load = self.load
        # shape: (batch, pomo)
        edge_mask = node_mask_to_edge_mask(self.ninf_mask_nodes, self.edge_to_node)
        self.step_state.ninf_mask = edge_mask
        # shape: (batch, pomo, E)

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        # tours: shape (B, P, N), edge indices
        problems_expanded = self.edge_attr.unsqueeze(1).expand(self.batch_size, self.pomo_size, self.edge_size, 2)
        _, _, n_steps = self.selected_edge_list.shape
        selected_edge_list_expanded = self.selected_edge_list.unsqueeze(-1).expand(self.batch_size, self.pomo_size, n_steps, 2)
        distances = torch.gather(problems_expanded, 2, selected_edge_list_expanded)
        # shape (B, P, N, 2)

        # Set all steps after finishing to zero distance
        finished_step_expanded = self.finished_step.unsqueeze(-1).unsqueeze(-1).expand(self.batch_size, self.pomo_size, n_steps, 2)
        distances = distances.masked_fill(torch.arange(n_steps).unsqueeze(0).unsqueeze(0).unsqueeze(-1) >= finished_step_expanded, 0.0)

        # Shape (B, P, 2)
        return torch.sum(distances, dim=2)
    
### Utils ###

def find_mask_indices_single(edge_to_node, selected_nodes):
    # edge_to_node: shape (B, E, 2)
    # selected_nodes: shape (B, P)
    B, E, _ = edge_to_node.shape
    _, P = selected_nodes.shape

    # Condition: Must go to the current node
    matches = (selected_nodes.unsqueeze(-1) == edge_to_node[:,:,1].unsqueeze(1))  # Shape (B, P, E)
    index_array = torch.arange(E, device=edge_to_node.device).expand(B, P, E)
    edges_to_mask = index_array[matches].reshape(B, P, -1)

    return edges_to_mask

def node_mask_to_edge_mask(node_mask, edge_to_node):
    B, P, N = node_mask.shape

    dest = edge_to_node[..., 1]
    _, E = dest.shape
    dest_idx = dest.unsqueeze(1).expand(B, P, E)

    # gather: result (B, P, E)
    edge_mask = torch.gather(node_mask, dim=2, index=dest_idx)
    return edge_mask