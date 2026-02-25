from dataclasses import dataclass
import torch

from MOCVRPProblemDef import get_random_problems


@dataclass
class Reset_State:
    dists: torch.Tensor = None # (B, problem, problem)
    demands: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, distribution, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.distribution = distribution

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.dists = None
        # shape: (batch, node, node)
        self.demands = None
        # shape: (B, N)

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem)
        self.ninf_mask = None
        # shape: (batch, pomo, problem)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size):
        self.batch_size = batch_size

        self.dists, self.demands = get_random_problems(self.distribution, batch_size, self.problem_size) # Shape (B, N, N), (B, N)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.dists = self.dists
        self.reset_state.demands = self.demands

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False 

        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.load = self.load

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.demands[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.load = self.load

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        # tours: shape (B, P, t)

        # Objective 1: total distance
        indices_from = self.selected_node_list[:, :, :-1]  # (B, P, t-1) - starting nodes of transitions
        indices_to = self.selected_node_list[:, :, 1:]     # (B, P, t-1) - ending nodes of transitions
        indices_from_last = self.selected_node_list[:, :, -1]  # (B, P) - last node of each tour
        indices_to_first = self.selected_node_list[:, :, 0]     # (B, P) - first node of each tour

        all_indices_from = torch.cat((indices_from, indices_from_last.unsqueeze(2)), dim=2)  # (B, P, t)
        all_indices_to = torch.cat((indices_to, indices_to_first.unsqueeze(2)), dim=2)  # (B, P, t)

        # Shape (B, P, t)
        from_distances = self.dists[torch.arange(self.batch_size).unsqueeze(1).unsqueeze(2), all_indices_from, all_indices_to]
        total_distances = from_distances.sum(dim=2)
        # (B, P)
    
        # Objective 2: makespans
        not_idx = (self.selected_node_list > 0)
        to_distances = from_distances.roll(shifts=1, dims=2)
        cum_lengths = torch.cumsum(to_distances, dim = 2)
      
        cum_lengths[not_idx] = 0
        sorted_cum_lengths, _ = cum_lengths.sort(axis = 2)
      
        rolled_sorted_cum_lengths = sorted_cum_lengths.roll(dims=2, shifts = 1)
        diff_mat = sorted_cum_lengths - rolled_sorted_cum_lengths
        diff_mat[diff_mat < 0] = 0
       
        makespans, _ = torch.max(diff_mat,dim = 2)
        # (B, P)

        objs = torch.stack([total_distances,makespans],axis = 2)
            
        # Shape (B, P, 2)
        return objs