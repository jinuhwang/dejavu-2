from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ReuseLoss(nn.Module):
    def __init__(
        self,
        target_reuse_rate: float,
        use_min_hloss: bool = False,
        use_min_sloss: bool = True,
        hloss_scaler: float = 0.0,
        sloss_scaler: float = 0.8,
        rloss_scaler: float = 1.0,
        max_reuse_per_layer: float = 1,
        rloss_duplicate_final_frame: bool = False,
    ):
        super().__init__()
        self.target_reuse_rate = target_reuse_rate
        self.use_min_hloss = use_min_hloss
        self.use_min_sloss = use_min_sloss

        if not isinstance(hloss_scaler, Iterable):
            hloss_scaler = [hloss_scaler]
        else:
            hloss_scaler = list(reversed(hloss_scaler))

        hloss_scaler = torch.tensor(hloss_scaler)
        self.register_buffer('hloss_scaler', hloss_scaler)

        self.sloss_scaler = sloss_scaler
        self.rloss_scaler = rloss_scaler
        self.max_reuse_per_layer = max_reuse_per_layer
        self.rloss_duplicate_final_frame = rloss_duplicate_final_frame

    def forward(
        self,
        hidden_states,
        output,
        original_hidden_states,
        original_output,
        reuse_maps,
    ):
        '''Drop support for mse loss and target similarity'''
        # Calculate mean cosine similarities of hidden states if hloss_scaler is not empty
        hidden_states = torch.stack(hidden_states[-len(self.hloss_scaler):], dim=0)
        original_hidden_states = torch.stack(original_hidden_states[-len(self.hloss_scaler):], dim=0)
        
        # [L, bsz, F, N]
        hidden_states_sim = F.cosine_similarity(hidden_states, original_hidden_states, dim=-1)

        # Calculate mean cosine similarities of output
        output_sim = F.cosine_similarity(output, original_output, dim=-1)

        # [bsz, num_stack, num_layers, num_tokens]
        reuse_rate_per_frame = reuse_maps.mean(dim=(-1, -2))

        reuse_rate_per_layer = reuse_maps.mean(dim=-1)
        reuse_rate_per_layer = torch.clamp(reuse_rate_per_layer, max=self.max_reuse_per_layer)
        reuse_rate_sum = reuse_rate_per_layer.sum()
        reuse_rate_numel = reuse_rate_per_layer.numel()

        if self.rloss_duplicate_final_frame:
            reuse_rate_sum += reuse_rate_per_frame[..., -1].sum()
            reuse_rate_numel += reuse_rate_per_frame[..., -1].numel()

        reuse_rate = reuse_rate_sum / reuse_rate_numel

        if self.use_min_hloss:
            # Find worst frame, then average over batch size and tokens
            hloss = 1 - hidden_states_sim.min(dim=2)[0].mean(dim=(-1, -2)) # [L]
        else:
            hloss = 1 - hidden_states_sim.mean(dim=(-1, -2, -3)) # [L]

        if self.use_min_sloss:
            sloss = 1 - output_sim.min(dim=-1)[0].mean()
        else:
            sloss = 1 - output_sim.mean()

        rloss = torch.relu(self.target_reuse_rate - reuse_rate)

        # We report final layer and second to last layer hidden state errors
        hidden_error = 1 - hidden_states_sim[-1].mean()
        if len(hidden_states_sim) > 1:
            hh_error = 1 - hidden_states_sim[-2].mean()
        else:
            hh_error = 0
        cls_error = 1 - output_sim.mean()

        loss = self.sloss_scaler*sloss + sum(self.hloss_scaler*hloss) + self.rloss_scaler*rloss

        return loss, hidden_error, hh_error, cls_error, reuse_rate, reuse_rate_per_frame
