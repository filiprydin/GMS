import torch
import torch.nn as nn

from Encoding.GREAT import GREATEncoder
from Decoding.MP_E_decoder import MPEDecoder

from torch_geometric.utils import to_dense_batch


class TSPModel(nn.Module):

    def __init__(self, encoder_params, decoder_params, env):
        super().__init__()
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

        self.env = env
        
        self.encoder = GREATEncoder(**encoder_params)
        self.encoded_edges = None # shape: (batch, emax * problem * (problem - 1), EMBEDDING_DIM)
        self.edge_indices = None # shape: (batch, emax * problem * (problem - 1), 2)

        self.decoder = MPEDecoder(**decoder_params)

    def pre_forward(self, reset_state, P, pref=None, encode=True):
        problems = reset_state.problems
        batch_size, self.n_edges, _ = problems.shape

        if encode: 
            encoded_edges_unbatched = self.encoder(problems, self.env.edge_to_node, self.env.problem_size)
            batch = torch.repeat_interleave(torch.arange(batch_size), self.n_edges)

            self.encoded_edges, _ = to_dense_batch(encoded_edges_unbatched, batch)
            # shape: (batch, E, EMBEDDING_DIM)

        self.decoder.reset(problems, self.encoded_edges, self.env.edge_to_node, P)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if self.env.selected_count == 0:
            self.last_edge_embedding = None

        probs, indices, embeddings = self.decoder(self.last_edge_embedding, state.current_node, state.ninf_mask)
        # probs: (B, P, E_out) probability of selecting edge between current and another node
        # indices: (B, P, E_out) corresponding next edges
        # embeddings: (B, P, E_out, H) corresponding edge embeddings

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

        # if using edge-based decoder, selected does not correspond to edges -> translate and get embedding of selected edge for next step
        _, _, E_out, H = embeddings.shape
        self.last_edge_embedding = torch.gather(embeddings, 2, selected.unsqueeze(2).unsqueeze(3).expand(batch_size, pomo_size, E_out, H))[:, :, 0, :]
        selected_edges = torch.gather(indices, 2, selected.unsqueeze(2).expand(batch_size, pomo_size, 1))[:, :, 0]
        
        return selected_edges, prob