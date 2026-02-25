import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

""" Edge-based version of Multi-pointer decoder """


class MPEDecoder(torch.nn.Module):
    def __init__(
        self,
        **model_params
    ):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.n_heads = self.model_params['head_num']
        self.tanh_clipping = self.model_params['logit_clipping']
        self.qkv_dim = self.model_params['qkv_dim']

        self.multi_pointer_level = 1
        self.add_more_query = True

        hyper_input_dim = 2
        hyper_hidden_embd_dim = 256
        self.embd_dim = 2
        self.hyper_output_dim = 6 * self.embd_dim
        
        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)
        
        self.hyper_Wq_first = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)
        self.hyper_Wq_visited = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)
        self.hyper_Wq_graph = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)

        self.hyper_Wq = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def assign(self, pref):
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        self.pref = pref
        
        self.Wq_first = self.hyper_Wq_first(mid_embd[:self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)
        self.Wq_last = self.hyper_Wq_last(mid_embd[self.embd_dim:2 * self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)
        self.Wq_visited = self.hyper_Wk(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)
        self.Wq_graph = self.hyper_Wk(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)

        self.Wq = self.hyper_Wk(mid_embd[4 * self.embd_dim: 5 * self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)
        self.Wk = self.hyper_Wk(mid_embd[5 * self.embd_dim: 6 * self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)

    def reset(self, dists, embeddings, indices, P):
        B, E, H = embeddings.shape

        if self.model_params["training_method"] == "Chb":
            self.dists = torch.maximum(self.pref[0] * dists[:, :, :, 0], self.pref[1] * dists[:, :, :, 1])
        elif self.model_params["training_method"] == "Linear":
            self.dists = self.pref[0] * dists[:, :, :, 0] + self.pref[1] * dists[:, :, :, 1]  # from (B, N, N, 2) to (B, N, N)
        self.embeddings = embeddings
        self.indices_group = indices.unsqueeze(1).expand(B, P, E, 2)
        self.embeddings_group = self.embeddings.unsqueeze(1).expand(B, P, E, H)
        
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)
        self.q_graph = F.linear(graph_embedding, self.Wq_graph)

        self.visited_embedding = torch.zeros(B, P, H)
        self.q_first = None

    def forward(self, last_edge_embedding, last_node_index, group_ninf_mask):
        B, E, H = self.embeddings.shape
        _, N, _ = self.dists.shape
        P = group_ninf_mask.size(1)

        with torch.no_grad():
            outgoing_indices = get_outgoing_indices(last_node_index, self.indices_group) # Get indices along the third dimension of outgoing edges (B, P, N-1)
            _, _, n_edges = outgoing_indices.shape

        outgoing_embeddings = torch.gather(self.embeddings_group, 2, outgoing_indices.unsqueeze(3).expand(B, P, n_edges, H)) # Corresponding embeddings (B, P, N-1, H)

        if self.n_heads > 1:
            logit_k = make_heads(
                F.linear(outgoing_embeddings, self.Wk), self.n_heads
            ).transpose(3, 4).transpose(1, 2)  # (B, n_heads, P, key_dim, N-1)
        else: 
            logit_k = outgoing_embeddings.transpose(2, 3) # (B, P, key_dim, N-1)

        # Get last edge embedding unless we are choosing node #2, in which case it is not defined
        if not last_edge_embedding is None: 
            q_last = F.linear(last_edge_embedding, self.Wq_last) # (B, P, H)
            self.visited_embedding = self.visited_embedding + last_edge_embedding / N # (B, P, H)
            q_visited = F.linear(self.visited_embedding, self.Wq_visited)

            if self.q_first is None:
                self.q_first = F.linear(last_edge_embedding, self.Wq_first) # (B, P, H)

        group_ninf_mask = group_ninf_mask.detach().clone() # (B, P, N)

        batch_indices = torch.arange(B).unsqueeze(1)
        distances = self.dists[batch_indices, last_node_index]  # (B, P, N)

        if self.q_first is None: # If second node is being chosen, only q_graph is defined
            final_q = self.q_graph.expand(B, P, H)
        else:
            if self.add_more_query:
                final_q = q_last + self.q_first + self.q_graph + q_visited
            else:
                final_q = q_last + self.q_first + self.q_graph

        with torch.no_grad():
            node_keys = torch.gather(self.indices_group, 2, outgoing_indices.unsqueeze(3).expand(B, P, n_edges, 2))[:, :, :, 1]
            node_keys = node_keys.to(dtype=torch.int64) # Which index in keys corresponds to visiting which node next, (B, P, N-1)
            distances_sorted = torch.gather(distances, 2, node_keys) # Get distances corresponding to edges that are being considered, (B, P, N-1)
        group_ninf_mask_edges = torch.gather(group_ninf_mask, 2, node_keys) # Get masks corresponding to edges that are being considered, (B, P, N-1)

        if self.n_heads > 1:
            final_q = make_heads_2(
                F.linear(final_q, self.Wq), self.n_heads
            )  # (B,n_head,P,q)
            score = (torch.einsum("bhpqn,bhpq->bhpn", logit_k, final_q) / math.sqrt(H)) - (
                distances_sorted / math.sqrt(2)
            ).unsqueeze(1)  # (B,n_head,P,N-1)
            if self.multi_pointer_level == 1:
                score_clipped = self.tanh_clipping * torch.tanh(score.mean(1))
            elif self.multi_pointer_level == 2:
                score_clipped = (self.tanh_clipping * torch.tanh(score)).mean(1)
            else:
                # add mask
                score_clipped = self.tanh_clipping * torch.tanh(score)
                mask_prob = group_ninf_mask_edges.detach().clone()
                mask_prob[mask_prob == -np.inf] = -1e8

                
                score_masked = score_clipped + mask_prob.unsqueeze(1)
                probs = F.softmax(score_masked, dim=-1).mean(1)
                return probs, node_keys, outgoing_embeddings
        else:
            final_q = F.linear(final_q, self.Wq)
            score = torch.einsum("bpqn,bpq->bpn", logit_k, final_q) / math.sqrt(
                H
            ) - distances_sorted / math.sqrt(2)
            score_clipped = self.tanh_clipping * torch.tanh(score)

        # add mask
        mask_prob = group_ninf_mask_edges.detach().clone()
        mask_prob[mask_prob == -np.inf] = -1e8
        score_masked = score_clipped + mask_prob
        probs = F.softmax(score_masked, dim=2)
        return probs, node_keys, outgoing_embeddings

### Utils ###

def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), qkv.size(2), n_heads, -1)
    return qkv.reshape(*shp).transpose(2, 3)

def make_heads_2(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)

def get_outgoing_indices(current_node_idx, indices):
    B, P, E, _ = indices.shape
    
    # Condition: Must start in the current node
    matches = (current_node_idx.unsqueeze(-1) == indices[:,:,:,0])  # Shape (B, P, E)
    index_array = torch.arange(E, device=indices.device).expand(B, P, E)
    edges_to_consider = index_array[matches].reshape(B, P, -1)

    return edges_to_consider