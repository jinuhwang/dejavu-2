import torch
from torch import nn
from .clip import CLIPVisionModelWithProjection

class DiffrateEncoderLayer(nn.Module):
    def __init__(
            self,
            original_encoder_layer,
            prune_kept_num,
            merge_kept_num,
            layer_idx=None,
        ):
        super().__init__()
        self.original_layer = original_encoder_layer
        self.prune_kept_num = prune_kept_num
        self.merge_kept_num = merge_kept_num

        # If layer_idx is not None, it passes spatial maps along with hidden states


    def forward(
            self,
            hidden_states,
            *args,
            output_attentions=False,
            output_qkvs=False,
            output_maps=False,
            **kwargs,
        ):
        B, N, dim = hidden_states.shape
        # Same as original encoder upto self attention
        residual = hidden_states

        hidden_states = self.original_layer.layer_norm1(hidden_states)
        hidden_states, attn_weights, qkvs, _ = self.original_layer.self_attn(
            hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=True,
            output_qkvs=output_qkvs,
        )
        hidden_states = residual + hidden_states # [B, N, dim]

        if output_maps:
            maps = torch.arange(N, device=hidden_states.device).unsqueeze(0).expand(B, N)
        else:
            maps = None

        if min(self.prune_kept_num, self.merge_kept_num) < N:
            # Size of attn_weights is [B, H, N, N]
            cls_attn = attn_weights[:, :, 0, 1:] # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)      # [B, N-1]
            _, idx = torch.sort(cls_attn, descending=True)
            cls_idx = torch.zeros((B, 1), device=idx.device, dtype=torch.long)
            idx = torch.cat((cls_idx, idx + 1), dim=1) # [B, N]

            if output_maps:
                # maps will be inverse indices that undo the sorting
                maps = torch.empty_like(idx)
                range_n = torch.arange(N, device=idx.device).unsqueeze(0).expand(B, N)
                maps.scatter_(1, idx, range_n)

            # Sort by attention weights
            hidden_states = torch.gather(
                hidden_states,
                dim=1,
                # Shape: [B, N] => [B, N, 1] => [B, N, dim]
                index=idx.unsqueeze(-1).expand(-1, -1, dim),
            )


            # [B, N, dim] => [B, prune_kept_num, dim]
            if self.prune_kept_num < N:
                hidden_states, idx, maps = self.prune(hidden_states, idx, maps)

            # [B, prune_kept_num, dim] => [B, merge_kept_num, dim]
            if self.merge_kept_num < hidden_states.shape[1]:
                hidden_states, maps = self.merge(hidden_states, idx, maps)

        # Rest is the same as original encoder
        residual = hidden_states
        hidden_states = self.original_layer.layer_norm2(hidden_states)
        hidden_states = self.original_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, attn_weights, qkvs, maps)

    def prune(self, hidden_states, sort_idx, maps):
        if maps is not None:
            mask_indices = sort_idx[:, self.prune_kept_num:]
            constant_tensor = torch.full_like(mask_indices, 512)
            maps.scatter_(1, mask_indices, constant_tensor)

        return hidden_states[:, :self.prune_kept_num], sort_idx[:, :self.prune_kept_num], maps

    def merge(self, hidden_states, sort_idx, maps, exclude_cls=True):
        B, _, dim = hidden_states.shape
        important_states = hidden_states[:, :self.merge_kept_num]
        unimportant_states = hidden_states[:, self.merge_kept_num:]

        normalized_states = hidden_states / hidden_states.norm(dim=-1, keepdim=True)
        normalized_important_states = normalized_states[:, :self.merge_kept_num]
        normalized_unimportant_states = normalized_states[:, self.merge_kept_num:]
        # [B, unimportant, dim] @ [B, dim, important] => [B, unimportant, important]
        similarity = normalized_unimportant_states @ normalized_important_states.transpose(-1, -2)
        if exclude_cls:
            similarity[..., 0] = -torch.inf

        # [B, unimportant, important] => [B, unimportant, 1]
        _, most_similar_idx = similarity.max(dim=-1, keepdim=True)

        important_states = important_states.scatter_reduce(
            dim=-2,
            index=most_similar_idx.expand(-1, -1, dim), # [B, unimportant, 1] => [B, unimportant, dim]
            src=unimportant_states,
            reduce='mean',
        )

        if maps is not None:
            unimportant_indices = sort_idx[:, self.merge_kept_num:]
            maps.scatter_(1, unimportant_indices, most_similar_idx.squeeze(-1))

        return important_states, maps

def create_diffrate_model(
    base_model_name,
    dataset,
    flops,
):
    from ...utils.paths import get_path_manager
    path_manager = get_path_manager()
    prune_kept_nums, merge_kept_nums = path_manager.get_diffrate_prune_merge(dataset, flops)

    clip = CLIPVisionModelWithProjection.from_pretrained(
        base_model_name,
        cache_dir=path_manager.paths['cache_dir'],
    )

    # Insert diffrate modules
    for layer_idx in range(len(clip.vision_model.encoder.layers)):
        prune = prune_kept_nums[layer_idx]
        merge = merge_kept_nums[layer_idx]

        original_encoder_layer = clip.vision_model.encoder.layers[layer_idx]
        clip.vision_model.encoder.layers[layer_idx] = DiffrateEncoderLayer(
            original_encoder_layer,
            prune,
            merge,
        )

    return clip



