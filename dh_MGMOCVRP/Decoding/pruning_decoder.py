import torch
import torch.nn.functional as F
import math

import torch.nn as nn

from torch_geometric.utils import softmax

class PruningDecoder(torch.nn.Module):
    def __init__(
        self,
        **model_params
    ):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.qkv_dim = self.model_params['qkv_dim']
        self.n_heads = self.model_params['head_num']

        hyper_input_dim = 2
        hyper_hidden_embd_dim = 256
        self.embd_dim = 2
        self.hyper_output_dim =  2*self.embd_dim

        self.tanh_clipping = self.model_params['logit_clipping']
        
        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)
        
        self.hyper_Wq = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(self.embd_dim, self.embedding_dim * self.n_heads * self.qkv_dim, bias=False)

    def assign(self, pref):
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        self.pref = pref
        
        self.Wq = self.hyper_Wq(mid_embd[:self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)
        self.Wk = self.hyper_Wk(mid_embd[self.embd_dim:2 * self.embd_dim]).reshape(self.embedding_dim, self.n_heads * self.qkv_dim)

    def reset(self, dists, edge_emb, edge_indices):
        # dists: (B * E, 2)

        self.dists = self.pref[0] * dists[:, 0] + self.pref[1] * dists[:, 1]  # from (B * E, 2) to (B * E)

        _, self.shared_nodes = torch.unique(edge_indices, dim=1, return_inverse=True)
        groups = torch.unique(self.shared_nodes)

        self.k = edge_emb

        group_sums = torch.zeros((groups.shape[0], edge_emb.shape[1]))
        group_sums.index_add_(0, self.shared_nodes, edge_emb)

        group_counts = torch.bincount(self.shared_nodes).clamp(min=1).unsqueeze(-1)
        group_means = group_sums / group_counts
        self.q = group_means[self.shared_nodes] # (B * E, Embedding_dim)

    def forward(self):
        final_q = F.linear(self.q, self.Wq).reshape(-1, self.n_heads, self.qkv_dim)
        logits_k = F.linear(self.k, self.Wk).reshape(-1, self.n_heads, self.qkv_dim)

        if self.model_params['only_by_distance']:
            score = -self.dists
        else: 
            score = torch.multiply(final_q, logits_k).sum(dim=2).mean(dim=1) / math.sqrt(self.embedding_dim) - self.dists / math.sqrt(2)

        score = self.tanh_clipping * torch.tanh(score)
        probs = softmax(score, self.shared_nodes)

        # probs: (B * E, 1)
        return probs.unsqueeze(1)
