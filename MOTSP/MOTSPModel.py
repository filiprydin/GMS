import torch
import torch.nn as nn

from Encoding.GREAT import GREATEncoder
from Encoding.Hybrid import HybridEncoder
from Decoding.MP_decoder import MPDecoder
from Decoding.MP_E_decoder import MPEDecoder

from torch_geometric.utils import to_dense_batch

class TSPModel(nn.Module):

    def __init__(self, encoder, decoder, encoder_params, decoder_params):
        super().__init__()
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        self.encoder_name = encoder
        self.decoder_name = decoder
        
        if encoder == "GREAT":
            self.encoder = GREATEncoder(return_nodes=True, **encoder_params)
            self.encoded_nodes = None # shape: (batch, problem, EMBEDDING_DIM)
        elif encoder == "GREAT-E":
            assert decoder == "MP-E" or decoder == "MP-E-R"
            self.encoder = GREATEncoder(return_nodes=False, **encoder_params)
            self.encoded_edges = None # shape: (batch, problem * (problem - 1), EMBEDDING_DIM)
            self.edge_indices = None # shape: (batch, problem * (problem - 1), 2)
        elif encoder == "hybrid":
            self.encoder = HybridEncoder(**encoder_params)
            self.encoded_nodes = None # shape: (batch, problem, EMBEDDING_DIM)
        
        if decoder == "MP":
            self.decoder = MPDecoder(**decoder_params)
        elif decoder == "MP-E":
            assert encoder == "GREAT-E"
            self.decoder = MPEDecoder(**decoder_params)

    def pre_forward(self, reset_state, P, pref=None, encode=True):
        problems = reset_state.problems
        batch_size, self.n_nodes, _, _ = problems.shape

        if self.encoder_name == "GREAT" and encode:
            encoded_nodes_unbatched = self.encoder(problems)
            batch = torch.repeat_interleave(torch.arange(batch_size), self.n_nodes)

            self.encoded_nodes, _ = to_dense_batch(encoded_nodes_unbatched, batch)
            # shape: (batch, problem, EMBEDDING_DIM)

        elif self.encoder_name == "GREAT-E" and encode:
            encoded_edges_unbatched, indices = self.encoder(problems)
            batch = torch.repeat_interleave(torch.arange(batch_size), self.n_nodes * (self.n_nodes - 1))

            self.encoded_edges, _ = to_dense_batch(encoded_edges_unbatched, batch)
            self.edge_indices = indices
            # shape: (batch, problem*(problem-1), EMBEDDING_DIM)

        elif self.encoder_name == "hybrid" and encode:
            encoded_nodes_unbatched = self.encoder(problems)
            batch = torch.repeat_interleave(torch.arange(batch_size), self.n_nodes)

            encoded_nodes_intermediate, _ = to_dense_batch(encoded_nodes_unbatched, batch)
            # shape: (batch, problem, EMBEDDING_DIM)

            self.encoded_nodes = self.encoder.forward_nodes(encoded_nodes_intermediate)

        if self.decoder_name == "MP":
            self.decoder.reset(problems, self.encoded_nodes, P)
        elif self.decoder_name == "MP-E":
            self.decoder.reset(problems, self.encoded_edges, self.edge_indices, P)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

            if self.encoder_name == "GREAT" or self.encoder_name == "hybrid":
                encoded_first_node = _get_encoding(self.encoded_nodes, selected)
                self.decoder.set_q1(encoded_first_node)
            elif self.encoder_name == "GREAT-E":
                self.last_edge_embedding = None
        else:
            if self.decoder_name == "MP":
                probs = self.decoder(state.current_node, state.ninf_mask)
                # shape: (batch, pomo, job)
            elif self.decoder_name == "MP-E":
                probs, indices, embeddings = self.decoder(self.last_edge_embedding, state.current_node, state.ninf_mask)
                # probs: (B, P, N-1) probability of selecting edge between current and another node
                # indices: (B, P, N-1) corresponding next nodes
                # embeddings: (B, P, N-1, H) corresponding edge embeddings

            if self.training or self.decoder_params['eval_type'] == 'softmax':
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

            if self.decoder_name == "MP-E":
                # if using edge-based decoder, selected does not correspond to nodes -> translate and get embedding of selected edge for next step
                _, _, n, H = embeddings.shape # Note n here is number of nodes - 1
                self.last_edge_embedding = torch.gather(embeddings, 2, selected.unsqueeze(2).unsqueeze(3).expand(batch_size, pomo_size, n, H))[:, :, 0, :]
                selected_nodes = torch.gather(indices, 2, selected.unsqueeze(2).expand(batch_size, pomo_size, n))[:, :, 0]
                
                return selected_nodes, prob
                
        return selected, prob
    
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
