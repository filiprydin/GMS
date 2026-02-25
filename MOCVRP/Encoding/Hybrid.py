import torch
import torch.nn as nn

from Encoding.GREAT import GREATEncoder
from Encoding.diff_attention import DiffAttentionEncoder

class HybridEncoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()

        self.model_params = model_params
        self.edge_attention_params = model_params['edge_attention_params']
        self.edge_attention_params['encoder_layer_num'] = model_params['L1']

        self.edge_attention = GREATEncoder(**self.edge_attention_params, return_nodes=True)

        self.node_attention_params = {}
        self.node_attention_params['encoder_layer_num'] = model_params['L2']
        self.node_attention_params['embedding_dim'] = self.edge_attention_params['embedding_dim']
        self.node_attention_params['qkv_dim'] = self.edge_attention_params['qkv_dim']
        self.node_attention_params['head_num'] = self.edge_attention_params['head_num']
        self.node_attention_params['ff_hidden_dim'] = self.edge_attention_params['ff_hidden_dim']

        self.node_attention = DiffAttentionEncoder(**self.node_attention_params)

    def forward(self, dists, demands):
        return self.edge_attention(dists, demands)
    
    def forward_nodes(self, encoded_nodes_intermediate, pref=None):
        return self.node_attention(encoded_nodes_intermediate)