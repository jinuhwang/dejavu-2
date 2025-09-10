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
        dloss_scaler: float = 0.0,
        hidden_loss_type: str = 'cosine',
        output_loss_type: str = 'cosine',
        detach_targets: bool = True,
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
        self.dloss_scaler = dloss_scaler
        self.hidden_loss_type = hidden_loss_type.lower()
        self.output_loss_type = output_loss_type.lower()
        self.detach_targets = detach_targets

    @staticmethod
    def _mse_tokenwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Memory‑friendly MSE per token/frame without materializing (x - y).

        x, y: (..., D) -> returns (...)
        """
        # y is expected to be detached by caller when needed
        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)
        xy = (x * y).sum(dim=-1)
        D = x.shape[-1]
        return (x2 + y2 - 2 * xy).div_(D)

    def forward(
        self,
        hidden_states,
        output,
        original_hidden_states,
        original_output,
        reuse_maps,
    ):
        """Compute reuse loss with cosine or MSE supervision.

        Shapes (typical):
        - hidden_states: List[L_i of] (B, F, N, D)
        - output: (B, F, D)
        - reuse_maps: (B, F, L, N)
        """

        # Optional detachment of targets
        if self.detach_targets:
            original_output = original_output.detach()
        # Determine how many layers contribute to hidden loss
        num_hlayers_cfg = int(len(self.hloss_scaler))
        L_avail = min(num_hlayers_cfg, len(hidden_states), len(original_hidden_states))
        hs_list = hidden_states[-L_avail:]
        ohs_list = original_hidden_states[-L_avail:]

        # Compute hidden-state loss per selected layer without stacking across layers
        hloss_vals = []  # list of scalars, length L_avail
        hidden_error = torch.tensor(0.0, device=output.device)
        hh_error = torch.tensor(0.0, device=output.device)

        for li, (hp, ht) in enumerate(zip(hs_list, ohs_list)):
            if self.detach_targets:
                ht = ht.detach()
            # hp/ht: (B, F, N, D)
            if self.hidden_loss_type == 'mse':
                # (B, F, N)
                token_err = self._mse_tokenwise(hp, ht)
                if self.use_min_hloss:
                    # Worst frame per sample/token, then mean
                    val = token_err.max(dim=1)[0].mean()
                else:
                    val = token_err.mean()
            elif self.hidden_loss_type == 'cosine':
                # (B, F, N)
                sim = F.cosine_similarity(hp, ht, dim=-1)
                if self.use_min_hloss:
                    val = 1.0 - sim.min(dim=1)[0].mean()
                else:
                    val = 1.0 - sim.mean()
            else:
                raise ValueError(f"Unknown hidden_loss_type={self.hidden_loss_type}")
            hloss_vals.append(val)

        if L_avail > 0:
            # Weights correspond to the last L layers (preserve order)
            w = self.hloss_scaler[-L_avail:].to(output.device)
            hloss = (torch.stack(hloss_vals) * w).sum()
            # Metrics on last and second-last layer for logging
            hp_last, ht_last = hs_list[-1], ohs_list[-1]
            if self.detach_targets:
                ht_last = ht_last.detach()
            if self.hidden_loss_type == 'mse':
                hidden_error = self._mse_tokenwise(hp_last, ht_last).mean()
            else:
                hidden_error = 1.0 - F.cosine_similarity(hp_last, ht_last, dim=-1).mean()
            if len(hs_list) > 1:
                hp_hh, ht_hh = hs_list[-2], ohs_list[-2]
                if self.detach_targets:
                    ht_hh = ht_hh.detach()
                if self.hidden_loss_type == 'mse':
                    hh_error = self._mse_tokenwise(hp_hh, ht_hh).mean()
                else:
                    hh_error = 1.0 - F.cosine_similarity(hp_hh, ht_hh, dim=-1).mean()
        else:
            hloss = torch.tensor(0.0, device=output.device)

        # Output supervision (pooled CLS per frame): (B, F, D) -> (B, F)
        if self.output_loss_type == 'mse':
            sloss_frame = self._mse_tokenwise(output, original_output)
            if self.use_min_sloss:
                sloss = sloss_frame.max(dim=-1)[0].mean()
            else:
                sloss = sloss_frame.mean()
            cls_error = sloss_frame.mean()
        elif self.output_loss_type == 'cosine':
            output_sim = F.cosine_similarity(output, original_output, dim=-1)
            if self.use_min_sloss:
                sloss = 1.0 - output_sim.min(dim=-1)[0].mean()
            else:
                sloss = 1.0 - output_sim.mean()
            cls_error = 1.0 - output_sim.mean()
        else:
            raise ValueError(f"Unknown output_loss_type={self.output_loss_type}")

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

        rloss = torch.relu(self.target_reuse_rate - reuse_rate)

        # Delta-consistency loss (layer-wise update consistency), computed pairwise to avoid stacking
        dloss = torch.tensor(0.0, device=output.device)
        if self.dloss_scaler > 0 and L_avail >= 2:
            pair_losses = []
            for (hp_prev, hp_next), (ht_prev, ht_next) in zip(
                zip(hs_list[:-1], hs_list[1:]), zip(ohs_list[:-1], ohs_list[1:])
            ):
                if self.detach_targets:
                    ht_prev = ht_prev.detach()
                    ht_next = ht_next.detach()
                delta_pred = hp_next - hp_prev
                delta_true = ht_next - ht_prev
                pair_losses.append((delta_true - delta_pred).pow(2).mean())
            if len(pair_losses) > 0:
                dloss = torch.stack(pair_losses).mean()

        loss = self.sloss_scaler*sloss + hloss + self.rloss_scaler*rloss + self.dloss_scaler*dloss

        # Keep API stable: place reuse_rate_per_frame last as before
        # New delta-consistency loss (dloss) is returned before it
        return loss, hidden_error, hh_error, cls_error, reuse_rate, dloss, reuse_rate_per_frame
