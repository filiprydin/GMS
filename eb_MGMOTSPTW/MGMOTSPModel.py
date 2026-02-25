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
        edge_attr = reset_state.edge_attr # (B, E, 2)
        edge_to_node = self.env.edge_to_node.to(dtype=torch.int64) # (B, E, w)
        tw_start = reset_state.tw_start # (B, N)
        tw_end = reset_state.tw_end # (B, N)
        batch_size, self.n_edges, _ = edge_attr.shape

        if encode:
            # Add time window as edge features
            tw_start_features = torch.gather(tw_start, 1, edge_to_node[:, :, 1]).unsqueeze(-1)
            tw_end_features = torch.gather(tw_end, 1, edge_to_node[:, :, 1]).unsqueeze(-1)
            # NOTE: In general one would include service time here too, we always put it to 0 and leave it out here
            edge_attr_aug = torch.cat((edge_attr, tw_start_features, tw_end_features), dim = 2)

            encoded_edges_unbatched = self.encoder(edge_attr_aug, edge_to_node, self.env.problem_size)
            batch = torch.repeat_interleave(torch.arange(batch_size), self.n_edges)

            self.encoded_edges, _ = to_dense_batch(encoded_edges_unbatched, batch)
            # shape: (batch, E, EMBEDDING_DIM)

        self.decoder.reset(edge_attr, self.encoded_edges, edge_to_node, P)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if self.env.selected_count == 0:
            # POMO -> create a temporary mask so that each sample goes to a separate node
            # Mask out every node except pomo node
            to_visit = torch.arange(start=1, end=pomo_size + 1)[None, :].expand(batch_size, -1) # (B, P)
            edges_to_visit = find_mask_indices(self.env.edge_to_node, to_visit)

            # Initialize all elements as negative infinity and set edges_to_visit to 0
            temp_ninf_mask = torch.zeros((batch_size, pomo_size, self.env.edge_size)) + float('-inf')
            batch_idx_expanded = state.BATCH_IDX[:, :, None].expand(batch_size, pomo_size, self.env.emax * (self.env.problem_size - 1))
            pomo_idx_expanded = state.POMO_IDX[:, :, None].expand(batch_size, pomo_size, self.env.emax * (self.env.problem_size - 1))
            temp_ninf_mask[batch_idx_expanded, pomo_idx_expanded, edges_to_visit] = 0

            ninf_mask = temp_ninf_mask # (batch, pomo, E)

            self.last_edge_embedding = None
        else:
            ninf_mask = state.ninf_mask

        probs, indices, embeddings = self.decoder(self.last_edge_embedding, state.current_node, ninf_mask, state.current_time)
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
    

### Utils ###

def find_mask_indices(edge_to_node, selected_nodes):
    B, E, _ = edge_to_node.shape
    _, P = selected_nodes.shape

    # Condition: Must go to the current node
    matches = (selected_nodes.unsqueeze(-1) == edge_to_node[:,:,1].unsqueeze(1))  # Shape (B, P, E)
    index_array = torch.arange(E, device=edge_to_node.device).expand(B, P, E)
    edges_to_mask = index_array[matches].reshape(B, P, -1)

    return edges_to_mask