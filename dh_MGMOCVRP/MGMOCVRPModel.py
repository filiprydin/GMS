import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoding.GREAT import GREATEncoder
from Encoding.diff_attention import DiffAttentionEncoder
from Decoding.MP_decoder import MPDecoder
from dh_MGMOCVRP.Decoding.pruning_decoder import PruningDecoder

from torch_geometric.utils import to_dense_batch


class CVRPModel(nn.Module):

    def __init__(self, encoder_params, head1_params, head2_params, env):
        super().__init__()
        self.encoder_params = encoder_params
        self.head1_params = head1_params
        self.head2_params = head2_params

        self.env = env
        
        self.encoder = GREATEncoder(**encoder_params)

        self.encoded_edges = None # shape: (batch * E, EMBEDDING_DIM)

        # Head 1 - sparsification
        head1_params['embedding_dim'] = encoder_params['embedding_dim']
        head1_params['qkv_dim'] = encoder_params['qkv_dim']
        head1_params['head_num'] = encoder_params['head_num']
        if self.head1_params['attention_layers'] > 0:
            head1_params['encoder_layer_num'] = head1_params['attention_layers']
            head1_params['dropout'] = encoder_params['dropout']
            head1_params['ff_hidden_dim'] = encoder_params['ff_hidden_dim']
            head1_params['initial_dim'] = encoder_params['embedding_dim']             

            self.head_1_attention = GREATEncoder(**head1_params)
        self.decoder_1 = PruningDecoder(**head1_params)

        # Head 2 - routing
        edge_to_node_params = {
            'embedding_dim': head2_params['embedding_dim'],
            'qkv_dim': head2_params['qkv_dim'],
            'head_num': head2_params['head_num'], 
            'ff_hidden_dim': head2_params['ff_hidden_dim'],
            'initial_dim': encoder_params['embedding_dim'],
            "dropout": encoder_params['dropout'], 
            'encoder_layer_num': 1, 
        }
        self.edge_to_node_emb = GREATEncoder(return_nodes=True, **edge_to_node_params)
        head2_params['encoder_layer_num'] = head2_params['attention_layers']
        self.head_2_attention = DiffAttentionEncoder(**head2_params)
        self.decoder_2 = MPDecoder(**head2_params)
    
    def encode(self):
        edge_attr = self.env.edge_attr # (B, E, 2)
        demands = self.env.demands
        edge_to_node = self.env.edge_to_node.to(dtype=torch.long) # (B, E, 2)

        demands_features = torch.gather(demands, 1, edge_to_node[:, :, 1]).unsqueeze(-1)
        edge_attr = torch.cat((edge_attr, demands_features), dim=2)

        self.edge_attr, self.edge_indices = format_encoder_input(edge_attr, edge_to_node, self.env.problem_size)
        self.encoded_edges = self.encoder(self.edge_attr, self.edge_indices, self.env.problem_size, emax=self.env.emax)
        # shape: (batch * E, EMBEDDING_DIM)

    def pre_forward_head_1(self, encode=True):

        if encode:
            encoded_edges = self.encoded_edges.detach().requires_grad_()
            
            if self.head1_params['attention_layers'] > 0 and not self.head1_params['only_by_distance']:
                self.edges_decoder_1 = self.head_1_attention(encoded_edges, self.edge_indices, self.env.problem_size, embed=False, emax=self.env.emax)
            else: 
                self.edges_decoder_1 = encoded_edges
            # (B * E, Embedding_dim)

        H = self.env.edge_attr.shape[2]
        dists = self.env.edge_attr.reshape(-1, H)  # (B * E, N)
        self.decoder_1.reset(dists, self.edges_decoder_1, self.edge_indices)

    def prune(self, samples_per_instance=1):

        probs = self.decoder_1()
        # shape: (B * E, 1)
        n_edges = probs.shape[0] // self.env.batch_size

        offset = torch.arange(samples_per_instance).view(1, -1) * self.env.problem_size * self.env.batch_size
        offset = offset.repeat_interleave(self.edge_indices.shape[1])
        self.edge_indices = self.edge_indices.repeat(1, samples_per_instance) + offset
        probs = probs.repeat(samples_per_instance, 1)
        # shape: (B * S * E, 1)

        selected_edges = self._sample_edges(probs, self.edge_indices)
        # (B * S * N * (N-1))
        if self.decoder_1.training or self.head1_params['eval_type'] == 'softmax':
            probs = probs[selected_edges].squeeze(1)
        else:
            probs = None

        # Expand to account for more samples
        self.encoded_edges = self.encoded_edges.repeat(samples_per_instance, 1)
        self.env.batch_size = self.env.batch_size * samples_per_instance

        # From (B * S * N * (N-1)) to (B * S, N * (N-1))
        if probs is not None: 
            probs = probs.view(self.env.batch_size, -1)
        selected_edges = selected_edges.view(self.env.batch_size, -1)
        offset = torch.arange(self.env.batch_size).view(-1, 1) * n_edges
        selected_edges = selected_edges - offset

        return selected_edges, probs
    
    def pre_forward_head_2(self, reset_state):
        
        selected_edges_unbatched = unbatch_edges(reset_state.selected_edges, self.env.edge_attr.shape[1]) # (B*N*(N-1))
        n_edges = len(selected_edges_unbatched)
        emb_dim = self.encoded_edges.shape[1]
        edges_decoder_2 = torch.gather(self.encoded_edges, 0, selected_edges_unbatched.unsqueeze(1).expand(n_edges, emb_dim)) # (B*N*(N-1), Embedding_dim)
        edge_to_node = torch.gather(self.edge_indices, 1, selected_edges_unbatched.unsqueeze(0).expand(2, n_edges)) # (2, B*N*(N-1))

        node_emb_unbatched = self.edge_to_node_emb(edges_decoder_2, edge_to_node, self.env.problem_size, embed=False, emax=1)
        
        batch = torch.repeat_interleave(torch.arange(self.env.batch_size), self.env.problem_size)
        node_emb, _ = to_dense_batch(node_emb_unbatched, batch) # (B, N, Embedding_dim)
        self.node_emb_decoder_2 = self.head_2_attention(node_emb) # (B, N, Embedding_dim)

        self.decoder_2.reset(reset_state.dist_matrix, self.node_emb_decoder_2, self.env.pomo_size)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

            encoded_first_node = _get_encoding(self.node_emb_decoder_2, selected)
            self.decoder_2.set_q1(encoded_first_node)
        elif state.selected_count == 1: # Second move, pomo
            selected = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, -1)
            prob = torch.ones(size=(batch_size, pomo_size))
        else:
            probs = self.decoder_2(state.current_node, state.ninf_mask, state.load)
            # shape: (batch, pomo, job)

            if self.decoder_2.training or self.head2_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)

                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)

                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None
                
        return selected, prob
    
    def _sample_edges(self, probs, edge_indices):
        # probs: (B * E, 1), edge_indices: (2, B * E)

        _, shared_nodes = torch.unique(edge_indices, dim=1, return_inverse=True)
        # (B * E)

        if self.decoder_1.training or self.head1_params['eval_type'] == 'softmax':
            sampled_indices = sample_indices(probs.squeeze(1), shared_nodes) # (B * N * (N - 1))

            return sampled_indices
        else: 
            max_prob_indices = get_max_prob_indices(probs.squeeze(1), shared_nodes) # (B * N  * (N - 1))

            return max_prob_indices

### UTILS ###

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes

def format_encoder_input(dists, indices, n_nodes):
    """
    For GREAT encoder 
    Converts (B, E, H) dists and (B, E, 2) indices to
    (B * E, H) dists and (2, B * E) indices
    """
    B, E, H = dists.shape  

    edge_attr = dists.reshape(-1, H)  # (B * E, N)

    offsets = torch.arange(B).view(-1, 1, 1) * n_nodes  # Shape: (B, 1, 1)
    indices = indices + offsets
    edge_index = indices.permute(2, 0, 1).reshape(2, -1)  # (2, B * E)

    return edge_attr, edge_index.to(dtype=torch.int64)

def unbatch_edges(indices, e_max):
    """
    Converts (B, N*(N-1)) indices to (B * N * (N-1)) indices
    """
    offsets = torch.arange(indices.shape[0]).view(-1, 1) * e_max  # Shape: (B, 1)
    indices = indices + offsets
    indices = indices.reshape(-1)  # (2, B * E)

    return indices.to(dtype=torch.int64)

def get_max_prob_indices(probs, groups):
    """
    Given list of probs and groups, returns the index of the highest prob for each group
    Used to retrieve which parallel edge has the highest probability
    """

    sorted_indices = torch.argsort(probs, descending=True)
    sorted_groups = groups[sorted_indices]

    unique, idx, counts = torch.unique(sorted_groups, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))

    first_group_indices = ind_sorted[cum_sum]

    max_prob_indices = sorted_indices[first_group_indices]
    max_prob_indices, _ = torch.sort(max_prob_indices) # Sort to preserve relative edge ordering
    return max_prob_indices

def sample_indices(probs, groups):
    """
    Given list of probs and groups, sample one index from each group
    Used to sample between parallel edges
    """

    # NOTE: we assume there are equally many edges between each node here. 
    unique_groups, inverse_indices = torch.unique(groups, return_inverse=True)
    num_groups = len(unique_groups)
    sorted_indices = torch.argsort(groups)
    sorted_probs = probs[sorted_indices]
    elements_per_group = len(probs) // num_groups
    reshaped_probs = sorted_probs.view(num_groups, elements_per_group)

    with torch.no_grad():
        selected = reshaped_probs.multinomial(1)
        # (num unique groups)

    offsets = torch.arange(selected.shape[0]).view(-1, 1) * elements_per_group
    selected = selected + offsets
    selected = selected.reshape(-1)

    sampled_indices = sorted_indices[selected]
    sampled_indices, _ = torch.sort(sampled_indices) # Sort to preserve relative edge ordering
    
    return sampled_indices