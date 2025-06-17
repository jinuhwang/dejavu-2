import torch
from torch import nn
import numpy as np
from .clip.modeling_clip import CLIPVisionModelWithProjection
from .modeling_siglip import SiglipVisionModel
from .blobnet import BlobNet
from .reuse import ReuseModule
from .reuse.decision import ReuseMLP
# from peft import get_peft_model, LoraConfig
from ...utils.paths import get_path_manager

path_manager = get_path_manager()
CACHE_DIR = path_manager.paths['cache_dir']

torch.manual_seed(42)

class ReuseEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            config,
            reuse_module,
            disable_reuse,
            original_encoder_layer,
            reuse_start='before_mlp',
        ):
        super().__init__()

        self.config = config

        token_num_per_side = self.config.image_size // config.patch_size
        kept_num = token_num_per_side**2

        self.disable_reuse = disable_reuse
        self.reuse_module = reuse_module

        self.layer_norm1 = original_encoder_layer.layer_norm1
        self.self_attn = original_encoder_layer.self_attn
        self.layer_norm2 = original_encoder_layer.layer_norm2
        self.mlp = original_encoder_layer.mlp
        self.reuse_start = reuse_start

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


    def forward(
            self,
            hidden_states,
            *args,
            pre_proj=None,
            attn_weights=None,
            cached_states=None,
            output_qkvs=False,
            compressed_map=None,
            ref_mask=None,
            ref_type=None,
            prev_reuse_map=None,
            position_embedding=None,
            **kwargs,
        ):
        bsz, N, dim = hidden_states.shape
        query_states, key_states, value_states, qkvs = self.layer_norm1_qkv_projection(hidden_states, output_qkvs=output_qkvs)

        if not self.disable_reuse and cached_states is not None:
            reuse_map, pre_proj, hidden_states, query_states, key_states, value_states = self.reuse_module(
                cached_states,
                pre_proj,
                hidden_states,
                query_states,
                key_states,
                value_states,
                attn_weights=attn_weights,
                compressed_map=compressed_map,
                ref_mask=ref_mask,
                ref_type=ref_type,
                prev_reuse_map=prev_reuse_map,
                position_embedding=position_embedding,
                **kwargs,
            )
        else:
            reuse_map = None

        if pre_proj is not None:
            cache_states = (pre_proj, hidden_states, query_states, key_states, value_states)
        else:
            # is first layer
            cache_states = None

        # MHSA
        num_heads = self.self_attn.num_heads
        head_dim = self.self_attn.head_dim
        proj_shape = (bsz * num_heads, -1, head_dim)
        # [4, B, N, dim] => [4*B, N, dim]
        q = query_states * self.self_attn.scale
        q = self.self_attn._shape(q, -1, bsz)
        q = q.view(*proj_shape)

        k = self.self_attn._shape(key_states, -1, bsz)
        k = k.view(*proj_shape)

        v = self.self_attn._shape(value_states, -1, bsz)
        v = v.view(*proj_shape)

        qk = torch.bmm(q, k.transpose(1, 2))

        # attn_weights: [4*B*H, N, N]
        attn_weights = torch.nn.functional.softmax(qk, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        attn_weights = attn_weights.view(bsz, num_heads, N, N)
        # [4*B*H, N, head_dim] => [4*B, H, N, head_dim]
        attn_output = attn_output.view(-1, self.self_attn.num_heads, N, head_dim)

        # [B, H, N, head_dim] => [B, N, H, head_dim]
        attn_output = attn_output.transpose(-3, -2)
        # [B, N, H, head_dim] => [B, N, dim(H*head_dim)]
        attn_output = attn_output.reshape(bsz, N, dim)
        pre_proj = attn_output
        attn_output = self.self_attn.out_proj(attn_output)

        # Add residual
        hidden_states = hidden_states + attn_output  # [B, N, dim]

        # Before FFN: residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # After FFN: hidden_states
        hidden_states = residual + hidden_states

        if self.reuse_start == 'before_qkv':
            # In this case, reuse module will check similarity using hidden states instead of pre_proj
            return hidden_states, cache_states, hidden_states, attn_weights, qkvs, reuse_map
        elif self.reuse_start == 'before_mlp':
            return pre_proj, cache_states, hidden_states, attn_weights, qkvs, reuse_map
        else:
            raise NotImplementedError(f'Unknown reuse_start: {self.reuse_start}')

# Sim model here caches result from previous frame
class ReuseModel(nn.Module):
    def __init__(
        self,
        base_model_name,
        reuse_modules,
        blobnet,
        dataset=None,
        cache_dir=CACHE_DIR,
        skip_last_layer_reuse=False,
        model_type='clip',
        input_dim=768,
    ):
        '''
        This model will forward frames in the following fixed pattern
        0 (no reuse)
        2 -> 0
        4 -> 4
        6 -> 4
        5 -> 4, 6
        '''
        super().__init__()

        if dataset == "msrvtt":
            base_model_name = path_manager.paths['msrvtt']['checkpoint']

        if model_type == 'clip':
            model_cls = CLIPVisionModelWithProjection
        elif model_type == 'siglip':
            model_cls = SiglipVisionModel
        else:
            raise NotImplementedError(f'Unknown model type: {model_type}')

        self.orig_model = model_cls.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
        )
        model = model_cls.from_pretrained(
            base_model_name,
            cache_dir=cache_dir,
        )

        config = model.config

        for layer_idx in range(len(model.vision_model.encoder.layers)):
            original_encoder_layer = model.vision_model.encoder.layers[layer_idx]

            disable_reuse=False
            if layer_idx == 0:
                disable_reuse = True
            elif skip_last_layer_reuse and layer_idx == len(model.vision_model.encoder.layers) - 1:
                disable_reuse = True

            model.vision_model.encoder.layers[layer_idx] = ReuseEncoderLayer(
                config,
                disable_reuse=disable_reuse,
                original_encoder_layer=original_encoder_layer,
                reuse_module=reuse_modules[layer_idx],
            )

        # Freeze all layers except thresholding module
        for param in model.parameters():
            param.requires_grad = False

        for param in self.orig_model.parameters():
            param.requires_grad = False

        token_num_per_side = config.image_size // config.patch_size
        # The blobnet have 6 channels
        # [mb_type, mv_x, mv_y, and three one_hot_frame_type]

        self.blobnet = blobnet

        for name, param in model.named_parameters():
            if 'reuse_module' in name or 'blobnet' in name:
                param.requires_grad = True

        self.model = model


    def forward_pre_encoder(
            self,
            pixel_values
        ):
        hidden_states = self.model.vision_model.embeddings(pixel_values)
        if hasattr(self.model.vision_model, 'pre_layrnorm'):
            hidden_states = self.model.vision_model.pre_layrnorm(hidden_states)

        return hidden_states

    def forward_post_encoder(
            self,
            hidden_states,
        ):
        hidden_states = self.model.vision_model.post_layernorm(hidden_states)
        if hasattr(self.model, 'visual_projection'):
            pooled_output = hidden_states[:, 0, :]
            image_embeds = self.model.visual_projection(pooled_output)
        elif hasattr(self.model.vision_model, 'head') and self.model.vision_model.use_head:
            image_embeds = self.model.vision_model.head(hidden_states)
        else:
            raise NotImplementedError

        return image_embeds

    def forward(
            self,
            pixel_values,
            *args,
            output_hidden_states=False,
            compressed=None,
            ref_mask=None,
            ref_type=None,
            **kwargs,
        ):
        B, F, *_ = pixel_values.shape
        pixel_values = pixel_values.view(-1, 3, 224, 224)
        hidden_states_list = self.forward_pre_encoder(pixel_values)

        _, N, dim = hidden_states_list.shape
        hidden_states_list = hidden_states_list.view(B, -1, N, dim)

        if output_hidden_states:
            ret_hidden_states = (hidden_states_list,)
        else:
            ret_hidden_states = None

        # [B, F, N, dim] => [F, B, N, dim]
        hidden_states_list = hidden_states_list.transpose(0, 1)
        
        B, F, C, T, H, W = compressed.shape
        compressed_input = compressed.view(B*F, -1, T, H, W)
        compressed_map = self.blobnet(compressed_input)
        compressed_map = compressed_map.view(B, F, -1)

        # (phqkv, B, frame_idx - 1, N, dim)
        prev_layer_reuse_maps = torch.zeros(B, F, N, device=hidden_states_list.device)

        embeddings = self.model.vision_model.embeddings
        position_embedding = embeddings.position_embedding(embeddings.position_ids)

        reuse_maps = []
        for layer_idx, encoder_layer in enumerate(self.model.vision_model.encoder.layers):
            cached_states = None
            next_pre_proj_list = []
            next_attn_weights_list = []
            next_hidden_states_list = []
            layer_reuse_maps = []

            for frame_idx in range(F):
                compressed = compressed_map[:, frame_idx]

                if layer_idx == 0:
                    pre_proj = None
                    attn_weights = None
                else:
                    pre_proj = pre_proj_list[frame_idx]
                    attn_weights = attn_weights_list[frame_idx]

                if ref_mask is None:
                    r = None
                else:
                    r = ref_mask[:, frame_idx, :frame_idx]

                pre_proj, cache_states, hidden_states, attn_weights, qkvs, reuse_map = encoder_layer(
                    hidden_states_list[frame_idx],
                    pre_proj=pre_proj,
                    attn_weights=attn_weights,
                    cached_states=cached_states if layer_idx != 0 else None,
                    compressed_map=compressed,
                    ref_mask=r,
                    ref_type=None if ref_type is None else ref_type[:, frame_idx],
                    prev_reuse_map=prev_layer_reuse_maps[:, frame_idx],
                    position_embedding=position_embedding,
                    **kwargs,
                )

                if cache_states is not None and frame_idx < F - 1:
                    if cached_states == None:
                        cached_states = cache_states
                    else:
                        cached_states = [torch.cat((cached, cache), dim=1) for (cached, cache) in zip(cached_states, cache_states)]

                next_pre_proj_list.append(pre_proj)
                next_attn_weights_list.append(attn_weights)
                next_hidden_states_list.append(hidden_states)

                if reuse_map is not None:
                    layer_reuse_maps.append(reuse_map)
                else:
                    layer_reuse_maps.append(torch.zeros(B, N, device=hidden_states.device))

            pre_proj_list = next_pre_proj_list
            attn_weights_list = next_attn_weights_list
            hidden_states_list = next_hidden_states_list
            if layer_idx != 0:
                prev_layer_reuse_maps = torch.stack(layer_reuse_maps, dim=1)
                reuse_maps.append(prev_layer_reuse_maps)

            if output_hidden_states:
                tmp_hidden_states_list = torch.stack(hidden_states_list, dim=1)
                ret_hidden_states += (tmp_hidden_states_list,)

        hidden_states_list = torch.stack(hidden_states_list, dim=1)
        hidden_states_list = hidden_states_list.view(-1, N, dim)
        outputs = self.forward_post_encoder(hidden_states_list)
        outputs = outputs.view(B, F, -1)

        reuse_maps = torch.stack(reuse_maps, dim=2)

        return  outputs, reuse_maps, ret_hidden_states

    def forward_eval(
            self,
            pixel_values,
            cached_states_prev=None,
            next_states_idx=None,
            *args,
            output_hidden_states=False,
            compressed=None,
            ref_mask=None,
            ref_type=None,
            **kwargs,
        ):
        B, F, *_ = pixel_values.shape
        pixel_values = pixel_values.view(-1, 3, 224, 224)
        hidden_states_list = self.forward_pre_encoder(pixel_values)

        _, N, dim = hidden_states_list.shape
        hidden_states_list = hidden_states_list.view(B, -1, N, dim)

        if output_hidden_states:
            ret_hidden_states = (hidden_states_list,)
        else:
            ret_hidden_states = None

        # [B, F, N, dim] => [F, B, N, dim]
        hidden_states_list = hidden_states_list.transpose(0, 1)
        
        if compressed is not None:
            B, F, C, T, H, W = compressed.shape
            compressed_input = compressed.view(B*F, -1, T, H, W)
            compressed_map = self.blobnet(compressed_input)
            compressed_map = compressed_map.view(B, F, -1)
        else:
            compressed_map = None

        # (phqkv, B, frame_idx - 1, N, dim)
        prev_layer_reuse_maps = torch.zeros(B, F, N, device=hidden_states_list.device)

        embeddings = self.model.vision_model.embeddings
        position_embedding = embeddings.position_embedding(embeddings.position_ids)

        reuse_maps = []

        cached_states_next = []
        for layer_idx, encoder_layer in enumerate(self.model.vision_model.encoder.layers):
            next_pre_proj_list = []
            next_attn_weights_list = []
            next_hidden_states_list = []
            layer_reuse_maps = []
            if layer_idx != 0 and cached_states_prev is not None:
                cached_states = cached_states_prev[layer_idx]
            else:
                cached_states = None

            for frame_idx in range(F):
                compressed = compressed_map[:, frame_idx]

                if layer_idx == 0:
                    pre_proj = None
                    attn_weights = None
                else:
                    pre_proj = pre_proj_list[frame_idx]
                    attn_weights = attn_weights_list[frame_idx]

                r = ref_mask[:, frame_idx, :frame_idx+1]

                pre_proj, cache_states, hidden_states, attn_weights, qkvs, reuse_map = encoder_layer(
                    hidden_states_list[frame_idx],
                    pre_proj=pre_proj,
                    attn_weights=attn_weights,
                    cached_states=cached_states if layer_idx != 0 else None,
                    compressed_map=compressed,
                    ref_mask=r,
                    ref_type=None if ref_type is None else ref_type[:, frame_idx],
                    prev_reuse_map=prev_layer_reuse_maps[:, frame_idx],
                    position_embedding=position_embedding,
                    **kwargs,
                )

                if next_states_idx is None:
                    # By default, we cache the last state (sequential reuse)
                    if frame_idx == F - 1:
                        cached_states_next.append(cache_states)
                elif frame_idx == next_states_idx:
                    cached_states_next.append(cache_states)

                if cache_states is not None and frame_idx < F - 1:
                    if cached_states == None:
                        cached_states = cache_states
                    cached_states = [torch.cat((cached, cache), dim=1) for (cached, cache) in zip(cached_states, cache_states)]

                next_pre_proj_list.append(pre_proj)
                next_attn_weights_list.append(attn_weights)
                next_hidden_states_list.append(hidden_states)

                if reuse_map is not None:
                    layer_reuse_maps.append(reuse_map)
                else:
                    layer_reuse_maps.append(torch.zeros(B, N, device=hidden_states.device))

            pre_proj_list = next_pre_proj_list
            attn_weights_list = next_attn_weights_list
            hidden_states_list = next_hidden_states_list
            if layer_idx != 0:
                prev_layer_reuse_maps = torch.stack(layer_reuse_maps, dim=1)
                reuse_maps.append(prev_layer_reuse_maps)

            if output_hidden_states:
                tmp_hidden_states_list = torch.stack(hidden_states_list, dim=1)
                ret_hidden_states += (tmp_hidden_states_list,)

        hidden_states_list = torch.stack(hidden_states_list, dim=1)
        hidden_states_list = hidden_states_list.view(-1, N, dim)
        outputs = self.forward_post_encoder(hidden_states_list)
        outputs = outputs.view(B, F, -1)

        reuse_maps = torch.stack(reuse_maps, dim=2)

        return  outputs, reuse_maps, ret_hidden_states, cached_states_next