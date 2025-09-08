from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from ..models.components.reusevit import ReuseModel


class InferenceReuseModel(torch.nn.Module):
    """
    Thin wrapper exposing the inference-optimized path of ReuseModel.

    - Uses `ReuseModel.forward_eval` for layer-wise scheduling.
    - Keeps architecture consistent with the training model (same parameters).
    - Optional Triton kernels are used indirectly via the underlying model if available.
    """

    def __init__(self, base_model: ReuseModel):
        super().__init__()
        self.base_model = base_model

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        compressed: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        ref_type: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        cached_states_prev: Optional[Any] = None,
        next_states_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Any]:
        return self.base_model.forward_eval(
            pixel_values=pixel_values,
            compressed=compressed,
            ref_mask=ref_mask,
            ref_type=ref_type,
            output_hidden_states=output_hidden_states,
            cached_states_prev=cached_states_prev,
            next_states_idx=next_states_idx,
            **kwargs,
        )

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        # Delegate directly to underlying model to guarantee architectural parity
        return self.base_model.load_state_dict(state_dict, strict=strict)

    @property
    def encoder_layers(self):
        return self.base_model.model.vision_model.encoder.layers

    @property
    def embed_dim(self) -> int:
        return self.base_model.model.visual_projection.out_features

