from dataclasses import dataclass
import torch

from MOTSPProblemDef import get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, problem, 2)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.distribution = env_params['distribution']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size):
        self.batch_size = batch_size

        self.problems = get_random_problems(self.distribution, batch_size, self.problem_size) # Shape (B, N, N, 2)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size))
        # shape: (batch, pomo, problem)

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
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
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

        # Shape (B, P, N, 2)
        from_distances = self.problems[torch.arange(self.batch_size).unsqueeze(1).unsqueeze(2), all_indices_from, all_indices_to]
        total_distances = from_distances.sum(dim=2)

        # Shape (B, P, 2)
        return total_distances