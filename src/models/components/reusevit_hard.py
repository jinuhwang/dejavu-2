#!/usr/bin/env python
from transformers import CLIPVisionConfig
import numpy as np

from .clip.modeling_optim_clip import CLIPVisionModelWithProjection
from ...utils.train import normalize_vector

from .reuse.decision import ReuseMLP
from .reuse.restoration import MLPRestoration, MergedMLPRestoration
from .stage_states import stage_states_local

# from kernl.model_optimization import optimize_model
import torch
from torch import nn

from ...utils.paths import get_path_manager
from .reusevit import ReuseEncoderLayer

path_manager = get_path_manager()
CACHE_DIR = path_manager.paths['cache_dir']

class OptimizedSimEncoderLayer(nn.Module):
    def __init__(
            self,
            config,
            reuse_module,
            original_encoder_layer,
            is_first_layer=False,
            is_last_layer=False,
        ):
        super().__init__()
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer

        if not is_first_layer:
            self.restoration_module = reuse_module.restoration_module
        if not is_last_layer:
            self.decision_module = reuse_module.decision_module

        self.layer_norm1 = original_encoder_layer.layer_norm1
        self.self_attn = original_encoder_layer.self_attn
        self.layer_norm2 = original_encoder_layer.layer_norm2
        self.mlp = original_encoder_layer.mlp

    # LayerNorm + Projection
    def layer_norm1_qkv_projection(
            self,
            hidden_states,
            output_qkvs=False,
        ):
        hidden_states = self.layer_norm1(hidden_states)

        # Projection
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_projected = self.self_attn.q_proj(hidden_states)
        key_projected = self.self_attn.k_proj(hidden_states)
        value_projected = self.self_attn.v_proj(hidden_states)

        if output_qkvs:
            qkvs = (
                query_projected.view(bsz, tgt_len, self.self_attn.num_heads, self.self_attn.head_dim),
                key_projected.view(bsz, tgt_len, self.self_attn.num_heads, self.self_attn.head_dim),
                value_projected.view(bsz, tgt_len, self.self_attn.num_heads, self.self_attn.head_dim)
            )
        else:
            qkvs = None

        return query_projected, key_projected, value_projected, qkvs

    # Self Attention
    def self_attn_pre_proj(
        self,
        hidden_states,
        query_projected,
        key_projected,
        value_projected,
        output_attentions=False,
    ):
        self_attn = self.self_attn
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = query_projected * self_attn.scale
        key_states = self_attn._shape(key_projected, -1, bsz)
        value_states = self_attn._shape(value_projected, -1, bsz)

        proj_shape = (bsz * self_attn.num_heads, -1, self_attn.head_dim)
        query_states = self_attn._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self_attn.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self_attn.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self_attn.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self_attn.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self_attn.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self_attn.num_heads, tgt_len, self_attn.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self_attn.num_heads, tgt_len, self_attn.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self_attn.num_heads, tgt_len, self_attn.head_dim)
        attn_output = attn_output.transpose(1, 2)
        pre_proj = attn_output.reshape(bsz, tgt_len, embed_dim)

        return pre_proj, attn_weights_reshaped

    def forward(
        self,
        list_hidden_states,
        *args,
        output_hidden_states=False,
        output_attentions=False,
        output_qkvs=False,
        reference_cache=None, # [B, N, dim]
        hqkv_cache=None,
        compressed_map=None,
        reference_type=None,
        disable_reuse=False,
        **kwargs,
    ):
        ''' Takes 4 frames as input
        args:
            list_hidden_states:
                1. First layer: Tensor of shape [4*B, N, dim]
                2. Other layers:
                  - list_gather_idxs: [4, B, N]
                  - list_hidden_states: [4, B, N, dim]
                  - diff_pre_proj: [4, B, N, dim]
        kwargs:
            cache: [5, B, 3, N, dim], where first dimension corresponds to H, Q, K, V, RefNorm
            is_first_four:
        '''
        hidden_states = None

        if not self.is_first_layer:
            # Gather index and compacted states received from previous layer
            # reference_states is a tensor after attention layer of previous layer
            list_gather_idxs, list_hidden_states, diff_pre_proj, reuse_map = list_hidden_states

        _, B, N, dim = hqkv_cache.shape
        h_cache, q_cache, k_cache, v_cache = hqkv_cache

        ##################
        # QKV Projection #
        ##################
        # Shape: [4*B, N, dim] or [1, N', dim]
        list_query_states, list_key_states, list_value_states, qkvs = self.layer_norm1_qkv_projection(
            list_hidden_states,
            output_qkvs=output_qkvs,
        )

        ###############
        # Restoration #
        ###############
        if not self.is_first_layer:
            # NOTE: we do have unexplored chance of reusing q*k value
            h_prime = torch.cat([h_cache.view(1, -1, dim), list_hidden_states], dim=1)
            q_prime = torch.cat([q_cache.view(1, -1, dim), list_query_states], dim=1)
            k_prime = torch.cat([k_cache.view(1, -1, dim), list_key_states], dim=1)
            v_prime = torch.cat([v_cache.view(1, -1, dim), list_value_states], dim=1)

            # prime: [1, N', dim] => [4, B, N', dim]
            # list_gather_idxs: [4, B, N] => [4, B, N, 1] => [4, B, N, dim]
            # result: [4, B, N, dim]
            list_hidden_states = torch.gather(
                h_prime.unsqueeze(0).expand(4, B, -1, -1),
                dim=2,
                index=list_gather_idxs.unsqueeze(-1).expand(-1, -1, -1, dim)
            )
            list_query_states = torch.gather(
                q_prime.unsqueeze(0).expand(4, B, -1, -1),
                dim=2,
                index=list_gather_idxs.unsqueeze(-1).expand(-1, -1, -1, dim)
            )
            list_key_states = torch.gather(
                k_prime.unsqueeze(0).expand(4, B, -1, -1),
                dim=2,
                index=list_gather_idxs.unsqueeze(-1).expand(-1, -1, -1, dim)
            )
            list_value_states = torch.gather(
                v_prime.unsqueeze(0).expand(4, B, -1, -1),
                dim=2,
                index=list_gather_idxs.unsqueeze(-1).expand(-1, -1, -1, dim)
            )

            # We compensate the differnce with MLP
            h_diff, q_diff, k_diff, v_diff = self.restoration_module.forward_mlp(diff_pre_proj[reuse_map])
            list_hidden_states[reuse_map] += h_diff
            list_query_states[reuse_map] += q_diff
            list_key_states[reuse_map] += k_diff
            list_value_states[reuse_map] += v_diff

            # We only update the cache for the first frame
            h_cache[:] = list_hidden_states[0]
            q_cache[:] = list_query_states[0]
            k_cache[:] = list_key_states[0]
            v_cache[:] = list_value_states[0]

            if output_hidden_states:
                hidden_states = list_hidden_states.clone()


        ##################
        # Self Attention #
        ##################
        pre_proj, attn_weights = self.self_attn_pre_proj(
            hidden_states=list_hidden_states.view(4*B, N, dim),
            query_projected=list_query_states.view(4*B, N, dim),
            key_projected=list_key_states.view(4*B, N, dim),
            value_projected=list_value_states.view(4*B, N, dim),
            output_attentions=True,
        )
        list_hidden_states = list_hidden_states.view(4, B, N, dim)
        attn_weights = attn_weights.view(4, B, -1, N, N)
        list_pre_proj = pre_proj.view(4, B, N, dim)

        # Final layer does not require reuse
        if self.is_last_layer:
            list_hidden_states = list_hidden_states[:, :, :1] # 4,B,1,dim
            pre_proj = list_pre_proj[:, :, :1] # 4,B,1,dim
            list_maps = None
        else:
            # Corresponds to 'cls^1' in training
            # Get rank of each token
            # [4, B, H, N, N] => [4, B, H, N-1]
            list_cls_attn = attn_weights[:, :, :, 0, 1:]
            # [4, B, H, N-1] => [4, B, N-1]
            importance = list_cls_attn.sum(dim=-2)

            compressed_map = compressed_map.view(4, B, N-1)

            # recompute_pre_proj: [1, N', dim]
            # diff_pre_proj: Used for restoration later [4, B, N, dim]
            # hidden_states: [1, N', dim]
            list_maps, list_gather_idxs, pre_proj, diff_pre_proj, list_hidden_states = self.stage_states(
                list_pre_proj,      # 4, B, N, dim
                reference_cache,
                list_hidden_states,
                importance=importance,
                compressed_map=compressed_map,
                reference_type=reference_type,
                disable_reuse=disable_reuse,
            )

        # Either [4*B, N, dim] or [1, N', dim]
        attn_output = self.self_attn.out_proj(pre_proj)
        list_hidden_states = list_hidden_states + attn_output

        residual = list_hidden_states
        list_hidden_states = self.layer_norm2(list_hidden_states)
        list_hidden_states = self.mlp(list_hidden_states)
        list_hidden_states = residual + list_hidden_states

        if not self.is_last_layer:
            list_hidden_states = (list_gather_idxs, list_hidden_states, diff_pre_proj, list_maps)

        return (list_hidden_states, attn_weights, qkvs, list_maps, hidden_states)

    def stage_states(
        self,
        list_pre_proj,      # [4, B, N, dim]
        reference_cache,    # [B, N, dim]
        list_hidden_states,
        importance,         # [4, B, N-1]
        compressed_map,     # [4, B, N, 1]
        reference_type,
        disable_reuse=False,
    ):
        _, B, N, dim = list_pre_proj.shape

        '''
        list_gather_idxs is used when gathering states, it assuems hidden states in the following format
                +---------------+-----------------------------------+
                |                 reference_cache                   |
                +---------------+-----------------------------------+
                | cached states |          compute_cache            |
                +---------------+--------------------+--------------+
        Batch 1 | cached states |     from frame 1   | from frame 2 | N'
                +---------------+--------------+--------------+-----+
        Batch 2 | cached states | from frame 1 | from frame 2 |       N''
                +---------------+--------------+--------------+
                |0             N|                  reference_cache_len^^^
        '''
        list_gather_idxs = torch.empty((4, B, N), dtype=torch.long, device=list_pre_proj.device)
        list_reuse_map = torch.empty((4, B, N), dtype=torch.bool, device=list_pre_proj.device)

        compute_cache_len = torch.zeros((1,), dtype=torch.long, device=list_pre_proj.device)
        compute_cache = torch.empty((B*4*N, dim), dtype=list_pre_proj.dtype, device=list_pre_proj.device)
        hidden_cache = torch.empty_like(compute_cache)

        diff_pre_proj = torch.empty((4, B, N, dim), dtype=list_pre_proj.dtype, device=list_pre_proj.device)

        list_pre_proj_norm = normalize_vector(list_pre_proj)      # 4, B, N, dim
        reference_states_norm = normalize_vector(reference_cache) # B, N, dim
        reference_states_gather_idx = torch.arange(B*N, dtype=torch.long, device=list_pre_proj.device).view(B, N)

        for frame_idx in range(4):
            if frame_idx == 0: # frame 4
                # [0]
                prev_ref = reference_cache
                prev_ref_norm = reference_states_norm # 0
                prev_ref_gather_idx = reference_states_gather_idx
                next_ref_norm = None
            elif frame_idx == 1: # frame 2
                prev_ref = reference_cache
                prev_ref_norm = reference_states_norm # 0
                prev_ref_gather_idx = reference_states_gather_idx
                next_ref = list_pre_proj[0]
                next_ref_norm = list_pre_proj_norm[0]     # 4
                next_ref_gather_idx = list_gather_idxs[0]
            elif frame_idx == 2: # frame 1
                prev_ref = reference_cache
                prev_ref_norm = reference_states_norm # 0
                prev_ref_gather_idx = reference_states_gather_idx
                next_ref = list_pre_proj[1]
                next_ref_norm = list_pre_proj_norm[1]     # 2
                next_ref_gather_idx = list_gather_idxs[1]
            elif frame_idx == 3: # frame 3
                prev_ref = list_pre_proj[1]
                prev_ref_norm = list_pre_proj_norm[1]     # 2
                prev_ref_gather_idx = list_gather_idxs[1]
                next_ref = list_pre_proj[0]
                next_ref_norm = list_pre_proj_norm[0]     # 4
                next_ref_gather_idx = list_gather_idxs[0]

            # [B*N, 1, dim] x [B*N, dim, 1] => [B*N, 1, 1]
            prev_similarity = torch.bmm(
                list_pre_proj_norm[frame_idx].view(B*N, 1, dim),
                prev_ref_norm.view(B*N, dim, 1)
            ).view(B, N)

            if next_ref_norm is not None:
                next_similarity = torch.bmm(
                    list_pre_proj_norm[frame_idx].view(B*N, 1, dim),
                    next_ref_norm.view(B*N, dim, 1)
                ).view(B, N)
                prev_more_similar = prev_similarity > next_similarity
                similarity = torch.where(prev_more_similar, prev_similarity, next_similarity)
                ref = torch.where(prev_more_similar.unsqueeze(-1), prev_ref, next_ref)
                ref_norm = torch.where(prev_more_similar.unsqueeze(-1), prev_ref_norm, next_ref_norm)
                ref_gather_idx = torch.where(prev_more_similar, prev_ref_gather_idx, next_ref_gather_idx)
            else:
                similarity = prev_similarity
                ref = prev_ref
                ref_norm = prev_ref_norm
                ref_gather_idx = prev_ref_gather_idx

            mlp_input = torch.cat(
                (
                    importance[frame_idx].unsqueeze(-1),
                    similarity[:, 1:].unsqueeze(-1),
                    compressed_map[frame_idx].unsqueeze(-1),
                    reference_type[frame_idx].unsqueeze(-2).expand(-1, N - 1, -1)
                ),
                dim=-1
            )
            decision = self.decision_module.forward_inner(mlp_input)
            # decision = decision.squeeze(-1)
            # reuse_map = decision > 0
            torch.gt(decision[..., 0], decision[..., 1], out=list_reuse_map[frame_idx, :, 1:])
            list_reuse_map[frame_idx, :, 0] = False # CLS token should not be reused
            if disable_reuse:
                list_reuse_map[frame_idx] = False

            stage_states_local(
                reuse_map=list_reuse_map[frame_idx],
                pre_proj=list_pre_proj[frame_idx],
                pre_proj_norm=list_pre_proj_norm[frame_idx],
                hidden_states=list_hidden_states[frame_idx],
                ref=ref,
                ref_norm=ref_norm,
                ref_gather_idx=ref_gather_idx,
                diff_pre_proj=diff_pre_proj[frame_idx],
                compute_cache=compute_cache,
                hidden_cache=hidden_cache,
                compute_cache_len=compute_cache_len,
                gather_idxs=list_gather_idxs[frame_idx],
            )

        # Only the first frame (frame 4) is passed on to next batch
        reference_cache[:] = list_pre_proj[0]

        compute_cache_len = compute_cache_len.item()
        recompute_pre_proj = compute_cache[:compute_cache_len].unsqueeze(0)
        list_hidden_states = hidden_cache[:compute_cache_len].unsqueeze(0)

        return list_reuse_map, list_gather_idxs, recompute_pre_proj, diff_pre_proj, list_hidden_states


class ReuseViTHard(nn.Module):
    def __init__(
        self,
        base_model_name,
        reuse_modules,
        blobnet,
        dataset=None,
        cache_dir=CACHE_DIR,
        skip_last_layer_reuse=False,
        **kwargs
    ):
        super().__init__()

        if dataset == "msrvtt":
            base_model_name = path_manager.paths['msrvtt']['checkpoint']

        model = CLIPVisionModelWithProjection.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
        )
        config = model.config

        # Replace encoder layers with reuse layers
        for layer_idx in range(len(model.vision_model.encoder.layers)):
            original_encoder_layer = model.vision_model.encoder.layers[layer_idx]
            model.vision_model.encoder.layers[layer_idx] = OptimizedSimEncoderLayer(
                config=config,
                reuse_module=reuse_modules[layer_idx],
                original_encoder_layer=original_encoder_layer,
                is_first_layer=layer_idx==0,
                is_last_layer=layer_idx == len(model.vision_model.encoder.layers) - 1,
            )

        self.blobnet = blobnet

        self.model = model.eval()

    def forward(self, *args, compressed=None, **kwargs):
        assert compressed is not None, "Compressed feature is required"

        B, F, C, T, H, W = compressed.shape
        compressed_input = compressed.view(B*F, -1, T, H, W)
        compressed_map = self.blobnet(compressed_input)
        compressed_map = compressed_map.view(B, F, -1)

        return self.model(*args, compressed_map=compressed_map, **kwargs)