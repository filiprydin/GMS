import math

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax

class GREATEncoder(nn.Module):
    """
    This class is a wrapper to built a GREAT-based encoder for a model trained with RL to solve TSP
    """

    def __init__(self, return_nodes, **model_params):
        super(GREATEncoder, self).__init__()

        self.model_params = model_params
        self.return_nodes = return_nodes

        assert (
            model_params["embedding_dim"] % model_params["head_num"] == 0
        ), "hidden_dimension must be divisible by the number of heads such that the dimension of the concatenation is equal to hidden_dim again"

        self.asymmetric = model_params["great_asymmetric"]
        self.dropout = nn.Dropout(p=model_params["dropout"])

        hidden_dim = model_params["embedding_dim"]
        hidden_dim_ff = model_params["ff_hidden_dim"]
        heads = model_params["head_num"]
        initial_dim = 2
        num_layers = model_params["encoder_layer_num"]

        self.embedder = Linear(initial_dim, hidden_dim)

        if self.asymmetric:
            self.att_layers = [
                GREATLayerAsymmetric(
                    hidden_dim, hidden_dim // heads, heads=heads, concat=True
                )
                for _ in range(num_layers)
            ]
        else:
            self.att_layers = [
                GREATLayer(
                    hidden_dim, hidden_dim // heads, heads=heads, concat=True
                )
                for _ in range(num_layers)
                ]
        self.att_layers = torch.nn.ModuleList(self.att_layers)

        self.ff_layers = [
            FFLayer(hidden_dim, hidden_dim_ff, hidden_dim) for _ in range(num_layers)
        ]
        self.ff_layers = torch.nn.ModuleList(self.ff_layers)

        ### Norms for layers
        self.att_Norms = torch.nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )
        self.ff_Norms = torch.nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, problems):
        dists, indices = _get_problem_dists(problems)
        _, num_nodes, _, _ = problems.shape
        edge_attr, edge_index = format_encoder_input(dists, indices, num_nodes)

        edges = self.embedder(edge_attr)  # E x hidden_dim

        for i, layer in enumerate(self.att_layers[:-1]):
            edge_index = edge_index.to(torch.int64)
            edges_agg = layer(
                edge_attr=edges, edge_index=edge_index, num_nodes=num_nodes
            )  # E x H
            edges_agg = self.dropout(edges_agg)
            edges = edges_agg + edges
            edges = self.att_Norms[i](edges)
            edge_ff = self.ff_layers[i](edges)
            edge_ff = self.dropout(edge_ff)
            edges = edge_ff + edges
            edges = self.ff_Norms[i](edges)

        layer = self.att_layers[
            -1
        ]  # get last layer which is special as we may want to return the node values

        if self.return_nodes:
            node_embeddings, edges_agg = layer(
                edge_attr=edges,
                edge_index=edge_index,
                num_nodes=num_nodes,
                return_nodes=True,
            )
            node_embeddings = self.att_Norms[-1](node_embeddings)
            node_embeddings_ff = self.ff_layers[-1](node_embeddings)
            node_embeddings = node_embeddings_ff + node_embeddings
            node_embeddings = self.ff_Norms[-1](node_embeddings)

            return node_embeddings
        else: 
            edge_index = edge_index.to(torch.int64)
            edges_agg = layer(
                edge_attr=edges, edge_index=edge_index, num_nodes=num_nodes
            )  # E x H
            edges_agg = self.dropout(edges_agg)
            edges = edges_agg + edges
            edges = self.att_Norms[i](edges)
            edge_ff = self.ff_layers[i](edges)
            edge_ff = self.dropout(edge_ff)
            edges = edge_ff + edges
            edges = self.ff_Norms[i](edges)
            return edges, indices
    
class FFLayer(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(FFLayer, self).__init__()
        self.linear1 = Linear(dim_in, dim_hid)
        self.linear2 = Linear(dim_hid, dim_out)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

class GREATLayer(MessagePassing):
    """
    A node-based GREAT layer for symmetric graphs (i.e., edge feature for edge (i,j) is THE SAME as for edge (j,i))
    """

    def __init__(self, dim_in, dim_out, heads=4, concat=True):
        super().__init__(node_dim=0)
        self.heads = heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.concat = concat

        self.lin_values = Linear(dim_in, self.heads * dim_out)
        self.lin_keys = Linear(dim_in, self.heads * dim_out)
        self.lin_queries = Linear(dim_in, self.heads * dim_out)

        if concat:
            self.edge_o = Linear(self.heads * dim_out * 2, self.heads * dim_out)
            self.lin_o = Linear(self.heads * dim_out, self.heads * dim_out)
        else:
            self.edge_o = Linear(dim_out * 2, dim_out)
            self.lin_o = Linear(dim_out, dim_out)

    def forward(self, edge_attr, edge_index, num_nodes, return_nodes=False):
        # edge_attr has shape [E, dim_in]
        # edge_index has shape [2, E]

        # x = torch.zeros((num_nodes, self.heads, self.dim_out)).float()
        edge_index = edge_index.to(torch.int64)
        out = self.propagate(
            edge_index, edge_attr=edge_attr
        )  # N x self.heads x self.dim_out

        if self.concat:
            out = out.view(-1, self.heads * self.dim_out)  # N x self.dim_out
            out = self.lin_o(out)  # N x self.dim_out
        else:
            out = out.mean(dim=1)  # N x self.dim_out
            out = self.lin_o(out)  # N x self.dim_out

        edge_agg = torch.cat(
            (out[edge_index[0]], out[edge_index[1]]), dim=1
        )  # E x self.dim_out * 2

        edge_agg = self.edge_o(edge_agg)  # E x self.dim_out

        if return_nodes:
            return out, edge_agg
        else:
            return edge_agg

    def message(self, edge_attr, index):
        h = self.heads
        d_out = self.dim_out

        values = self.lin_values(edge_attr)  # E x dim_out * h
        values = values.view(-1, h, d_out)  # E x h x dim_out

        queries = self.lin_queries(edge_attr)  # E x dim_out * h
        queries = queries.view(-1, h, d_out)  # E x h x dim_out

        keys = self.lin_keys(edge_attr)  # E x dim_out * h
        keys = keys.view(-1, h, d_out)  # E x h x dim_out

        alpha = (queries * keys).sum(dim=-1) / math.sqrt(self.dim_out)
        index = index.to(torch.int64)
        alpha = softmax(alpha, index)

        out = values * alpha.view(-1, self.heads, 1)

        return out
    
class GREATLayerAsymmetric(MessagePassing):
    """
    A node-based GREAT layer for asymmetric graphs (i.e., edge feature for edge (i,j) is NOT THE SAME as for edge (j,i))
    """

    def __init__(self, dim_in, dim_out, heads=4, concat=True):
        super().__init__(node_dim=0)
        self.heads = heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.concat = concat

        # Outgoing edges
        self.lin_values_out = Linear(dim_in, self.heads * dim_out)
        self.lin_keys_out = Linear(dim_in, self.heads * dim_out)
        self.lin_queries_out = Linear(dim_in, self.heads * dim_out)

        # Ingoing edges
        self.lin_values_in = Linear(dim_in, self.heads * dim_out)
        self.lin_keys_in = Linear(dim_in, self.heads * dim_out)
        self.lin_queries_in = Linear(dim_in, self.heads * dim_out)

        if concat:
            self.edge_o = Linear(self.heads * dim_out * 2, self.heads * dim_out)
            self.lin_o = Linear(self.heads * dim_out * 4, self.heads * dim_out)
        else:
            self.edge_o = Linear(dim_out * 2, dim_out)
            self.lin_o = Linear(dim_out * 4, dim_out)

    def forward(self, edge_attr, edge_index, num_nodes, return_nodes=False):
        # edge_attr has shape [E, dim_in]
        # edge_index has shape [2, E]

        # x = torch.zeros((num_nodes, self.heads, self.dim_out)).float()
        out = self.propagate(
            edge_index, edge_attr=edge_attr
        )  # N x self.heads x self.dim_out

        if self.concat:
            out = out.view(-1, self.heads * self.dim_out * 4)  # N x self.dim_out * 2
            out = self.lin_o(out)  # N x self.dim_out
        else:
            out = out.mean(dim=1)  # N x self.dim_out * 2
            out = self.lin_o(out)  # N x self.dim_out

        edge_agg = torch.cat(
            (out[edge_index[0]], out[edge_index[1]]), dim=1
        )  # E x self.dim_out * 2

        edge_agg = self.edge_o(edge_agg)  # E x self.dim_out

        if return_nodes:
            return out, edge_agg
        else:
            return edge_agg

    def message(self, edge_attr, edge_index):
        h = self.heads
        d_out = self.dim_out

        # ingoing edges
        values_in = self.lin_values_in(
            edge_attr
        )  # E x temp * h; temp = dim_out if concat = False, else dim_out//h
        values_in = values_in.view(-1, h, d_out)  # E x h x temp

        queries_in = self.lin_queries_in(edge_attr)  # E x temp * h
        queries_in = queries_in.view(-1, h, d_out)  # E x h x temp

        keys_in = self.lin_keys_in(edge_attr)  # E x temp * h
        keys_in = keys_in.view(-1, h, d_out)  # E x h x temp

        alpha_in = (queries_in * keys_in).sum(dim=-1) / math.sqrt(self.dim_out)
        alpha_in = softmax(alpha_in, edge_index[1])  # E x h

        out_in = values_in * alpha_in.view(-1, self.heads, 1)  # E x h x temp

        #  outgoing edges
        values_out = self.lin_values_out(edge_attr)  # E x temp * h
        values_out = values_out.view(-1, h, d_out)  # E x h x temp

        queries_out = self.lin_queries_out(edge_attr)  # E x temp * h
        queries_out = queries_out.view(-1, h, d_out)  # E x h x temp

        keys_out = self.lin_keys_out(edge_attr)  # E x temp * h
        keys_out = keys_out.view(-1, h, d_out)  # E x h x temp

        alpha_out = (queries_out * keys_out).sum(dim=-1) / math.sqrt(self.dim_out)
        alpha_out = softmax(alpha_out, edge_index[0])  # E x h

        out_out = values_out * alpha_out.view(-1, self.heads, 1)  # E x h x temp

        # Concatenate four terms -> find a reverse edge for each edge 
        # NOTE: This assumes an edge is immediately followed by its reverse counterpart, this assumption can be relaxed
        out_out_swapped = out_out.clone()
        out_out_swapped[::2], out_out_swapped[1::2] = out_out[1::2].clone(), out_out[::2].clone()
        out_in_swapped = out_in.clone()
        out_in_swapped[::2], out_in_swapped[1::2] = out_in[1::2].clone(), out_in[::2].clone()
        out = torch.cat((out_in, out_in_swapped, out_out, out_out_swapped), dim=2)

        return out
    
### UTILS ###

def format_encoder_input(reset_state_dists, reset_state_indices, n_nodes):
    """For GREAT encoder """
    B, E, N = reset_state_dists.shape  

    edge_attr = reset_state_dists.reshape(-1, N)  # (B * E, N)

    offsets = torch.arange(B).view(-1, 1, 1) * n_nodes  # Shape: (B, 1, 1)
    reset_state_indices = reset_state_indices + offsets
    edge_index = reset_state_indices.permute(2, 0, 1).reshape(2, -1)  # (2, B * E)

    return edge_attr, edge_index

def _get_problem_dists(problems):
    """ 
    Converts set of distance matrices (B, N, N, Nobj) to edge attributes (B, E, Nobj) and edge indices (B, E, 2)
    """
    B, N, _, Nobj = problems.shape

    idx_i, idx_j = torch.triu_indices(N, N, offset=1)
    idx_pairs = torch.stack([idx_i, idx_j], dim=1)  # shape (num_edges, 2)
    reverse_idx_pairs = torch.stack([idx_j, idx_i], dim=1)  # shape (num_edges, 2)
    all_idx_pairs = torch.cat([idx_pairs, reverse_idx_pairs], dim=1).view(-1, 2)  # Flatten into pairs
    indices = all_idx_pairs.unsqueeze(0).repeat(B, 1, 1).to(torch.int32)
    i_indices, j_indices = all_idx_pairs[:, 0], all_idx_pairs[:, 1]

    dists = problems[:, i_indices, j_indices, :]

    return dists, indices