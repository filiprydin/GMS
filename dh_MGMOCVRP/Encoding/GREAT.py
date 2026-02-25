import math

import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax

class GREATEncoder(nn.Module):
    """
    This class is a wrapper to build a GREAT-based encoder for a model trained with RL to solve TSP
    """

    def __init__(self, return_nodes = False, **model_params):
        super(GREATEncoder, self).__init__()

        self.model_params = model_params
        self.return_nodes = return_nodes

        assert (
            model_params["embedding_dim"] % model_params["head_num"] == 0
        ), "hidden_dimension must be divisible by the number of heads such that the dimension of the concatenation is equal to hidden_dim again"

        self.dropout = nn.Dropout(p=model_params["dropout"])

        hidden_dim = model_params["embedding_dim"]
        hidden_dim_ff = model_params["ff_hidden_dim"]
        heads = model_params["head_num"]
        initial_dim = 3
        num_layers = model_params["encoder_layer_num"]

        self.embedder = Linear(initial_dim, hidden_dim)

        self.att_layers = [
            GREATLayerAsymmetric(
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

    def forward(self, edge_attr, edge_index, num_nodes, embed=True, emax=1):
        # problems: (B * E, Input_dim)
        # edge_index: (2, B * E)

        if embed:
            edges = self.embedder(edge_attr)  # E x hidden_dim
        else:
            edges = edge_attr

        if not self.return_nodes:
            for i, layer in enumerate(self.att_layers):
                edge_index = edge_index.to(torch.int64)
                edges_agg = layer(
                    edge_attr=edges, edge_index=edge_index, num_nodes=num_nodes, emax=emax
                )  # E x H
                edges_agg = self.dropout(edges_agg)
                edges = edges_agg + edges
                edges = self.att_Norms[i](edges)
                edge_ff = self.ff_layers[i](edges)
                edge_ff = self.dropout(edge_ff)
                edges = edge_ff + edges
                edges = self.ff_Norms[i](edges)

            return edges
        else:
            for i, layer in enumerate(self.att_layers[:-1]):
                edge_index = edge_index.to(torch.int64)
                edges_agg = layer(
                    edge_attr=edges, edge_index=edge_index, num_nodes=num_nodes, emax=emax
                )  # E x H
                edges_agg = self.dropout(edges_agg)
                edges = edges_agg + edges
                edges = self.att_Norms[i](edges)
                edge_ff = self.ff_layers[i](edges)
                edge_ff = self.dropout(edge_ff)
                edges = edge_ff + edges
                edges = self.ff_Norms[i](edges)

            layer = self.att_layers[-1]

            node_embeddings, _ = layer(
                edge_attr=edges, edge_index=edge_index, num_nodes=num_nodes, return_nodes=True, emax = emax
            )
            node_embeddings = self.att_Norms[-1](node_embeddings)
            node_embeddings_ff = self.ff_layers[-1](node_embeddings)
            node_embeddings = node_embeddings_ff + node_embeddings
            node_embeddings = self.ff_Norms[-1](node_embeddings)

            return node_embeddings
    
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

    def forward(self, edge_attr, edge_index, num_nodes, return_nodes=False, emax = 1):
        # edge_attr has shape [E, dim_in]
        # edge_index has shape [2, E]
        self.emax = emax

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
        # NOTE: We assume all parallel edges are bundled together and immediately followed by their reverse counterpart. 
        # Also assume equally many edges in both directions. 
        out_out_swapped = out_out.clone()
        out_in_swapped = out_in.clone()

        switch_int = 2 * self.emax

        for i in range(self.emax):
            idx1 = i
            idx2 = self.emax + i

            out_out_swapped[idx1::switch_int], out_out_swapped[idx2::switch_int] = out_out[idx2::switch_int].clone(), out_out[idx1::switch_int].clone()
            out_in_swapped[idx1::switch_int], out_in_swapped[idx2::switch_int] = out_in[idx2::switch_int].clone(), out_in[idx1::switch_int].clone()

        out = torch.cat((out_in, out_in_swapped, out_out, out_out_swapped), dim=2)

        return out