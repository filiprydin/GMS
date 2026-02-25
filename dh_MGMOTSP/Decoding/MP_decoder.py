import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


"""Code by https://github.com/Pointerformer/Pointerformer """


class MPDecoder(torch.nn.Module):
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

    def reset(self, dists, embeddings, P):
        B, N, H = embeddings.shape

        if self.model_params["training_method"] == "Chb":
            self.dists = torch.maximum(self.pref[0] * dists[:, :, :, 0], self.pref[1] * dists[:, :, :, 1])
        elif self.model_params["training_method"] == "Linear":
            self.dists = self.pref[0] * dists[:, :, :, 0] + self.pref[1] * dists[:, :, :, 1]  # from (B, N, N, 2) to (B, N, N)
        self.embeddings = embeddings
        self.embeddings_group = self.embeddings.unsqueeze(1).expand(B, P, N, H)
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = F.linear(graph_embedding, self.Wq_graph)
        self.logit_k = embeddings.transpose(1, 2)
        if self.n_heads > 1:
            self.logit_k = make_heads(
                F.linear(embeddings, self.Wk), self.n_heads
            ).transpose(2, 3)  # [B, n_heads, key_dim, N]

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        self.q_first = F.linear(encoded_q1, self.Wq_first)

    def forward(self, last_node, group_ninf_mask):
        B, N, H = self.embeddings.shape
        P = group_ninf_mask.size(1)

        # Get last node embedding
        last_node_index = last_node.view(B, P, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = F.linear(last_node_embedding, self.Wq_last)

        group_ninf_mask = group_ninf_mask.detach()

        mask_visited = group_ninf_mask.clone()
        mask_visited[mask_visited == -np.inf] = 1.0
        q_visited = F.linear(torch.bmm(mask_visited, self.embeddings) / N, self.Wq_visited)

        batch_indices = torch.arange(B).unsqueeze(1)
        distances = self.dists[batch_indices, last_node]  # B x G x N

        if self.add_more_query:
            final_q = q_last + self.q_first + self.q_graph + q_visited
        else:
            final_q = q_last + self.q_first + self.q_graph

        if self.n_heads > 1:
            final_q = make_heads(
                F.linear(final_q, self.Wq), self.n_heads
            )  # (B,n_head,G,H)  (B,n_head,H,N)
            score = (torch.matmul(final_q, self.logit_k) / math.sqrt(H)) - (
                distances / math.sqrt(2)
            ).unsqueeze(1)  # (B,n_head,G,N)
            if self.multi_pointer_level == 1:
                score_clipped = self.tanh_clipping * torch.tanh(score.mean(1))
            elif self.multi_pointer_level == 2:
                score_clipped = (self.tanh_clipping * torch.tanh(score)).mean(1)
            else:
                # add mask
                score_clipped = self.tanh_clipping * torch.tanh(score)
                mask_prob = group_ninf_mask.detach().clone()
                mask_prob[mask_prob == -np.inf] = -1e8

                score_masked = score_clipped + mask_prob.unsqueeze(1)
                probs = F.softmax(score_masked, dim=-1).mean(1)
                return probs
        else:
            score = torch.matmul(final_q, self.logit_k) / math.sqrt(
                H
            ) - distances / math.sqrt(2)
            score_clipped = self.tanh_clipping * torch.tanh(score)

        # add mask
        mask_prob = group_ninf_mask.detach().clone()
        mask_prob[mask_prob == -np.inf] = -1e8
        score_masked = score_clipped + mask_prob
        probs = F.softmax(score_masked, dim=2)

        return probs

def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)
