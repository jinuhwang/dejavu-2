import torch
from torch import nn
from torch.nn import functional as F
from .similarity import CosineSimilarity, HeadwiseCosineSimilarity, L2Similarity, SADSimilarity, LocalL2Similarity, LocalCosineSimilarity
from .restoration import PassthroughRestoration, DiffRestoration, MLPRestoration
from .importance import CLSImportance, TldrImportance, ZeroTPruenImportance, NoneImportance
from .decision import ReuseMLP, TopKDecision
from .gating import SteepSigmoid, GumbelSoftmaxGating, HardGating, AdafuseGating


class ReuseModule(nn.Module):
    def __init__(
            self,
            decision,
            similarity,
            importance,
            gating,
            restoration,
            skip_cls=True,
        ):
        super().__init__()

        self.similarity_module = similarity
        self.importance_module = importance
        self.decision_module = decision
        self.gating_module = gating
        self.restoration_module = restoration

        self.skip_cls = skip_cls


    def forward(
            self,
            cached_states,
            pre_proj,
            hidden_states,
            query_states,
            key_states,
            value_states,
            attn_weights,
            compressed_map=None,
            ref_mask=None,
            ref_type=None,
            prev_reuse_map=None,
            position_embedding=None,
            **kwargs
        ):
        B, N, dim = hidden_states.shape

        (
            cached_pre_proj,
            cached_hidden_states,
            cached_query_states,
            cached_key_states,
            cached_value_states,
        ) = cached_states

        if self.skip_cls:
            cls_pre_proj = pre_proj[:, 0:1]
            cls_hidden_states = hidden_states[:, 0:1]
            cls_query_states = query_states[:, 0:1]
            cls_key_states = key_states[:, 0:1]
            cls_value_states = value_states[:, 0:1]

            pre_proj = pre_proj[:, 1:]
            hidden_states = hidden_states[:, 1:]
            query_states = query_states[:, 1:]
            key_states = key_states[:, 1:]
            value_states = value_states[:, 1:]
            if prev_reuse_map is not None:
                prev_reuse_map = prev_reuse_map[:, 1:]
            lhs_N = N - 1
        else:
            lhs_N = N

        similarity, norm = self.similarity_module(pre_proj, cached_pre_proj, position_embedding=position_embedding)

        assert ref_mask is not None, "Mask must be provided for this model"
        similarity = similarity.view(B, lhs_N, -1, N)
        ref_mask = ref_mask[:, :similarity.shape[2]]
        ref_mask = ref_mask.view(B, 1, -1, 1).expand(-1, lhs_N, -1, N)
        similarity[~ref_mask] = -1e9
        similarity = similarity.view(B, lhs_N, -1)

        importance = self.importance_module(attn_weights)
        reuse_decision, most_similar_idx, _ = self.decision_module(
            importance=importance,
            similarity=similarity if norm is None else (similarity, norm),
            compressed_map=compressed_map,
            ref_type=ref_type,
            prev_reuse_map=prev_reuse_map,
        )

        # Gather most similar inputs
        most_similar_pre_proj = torch.gather(
            cached_pre_proj,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_hidden_states = torch.gather(
            cached_hidden_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_query_states = torch.gather(
            cached_query_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_key_states = torch.gather(
            cached_key_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_value_states = torch.gather(
            cached_value_states,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        (
            restored_hidden_states,
            restored_query_states,
            restored_key_states,
            restored_value_states,
        ) = self.restoration_module(
            current_pre_proj=pre_proj,
            most_similar_pre_proj=most_similar_pre_proj,
            most_similar_hidden_states=most_similar_hidden_states,
            most_similar_query_states=most_similar_query_states,
            most_similar_key_states=most_similar_key_states,
            most_similar_value_states=most_similar_value_states,
        )

        hard = kwargs.get('hard', False)
        tau = kwargs.get('tau', None)
        
        reuse_map, (
            pre_proj,
            hidden_states,
            query_states,
            key_states,
            value_states,
        ) = self.gating_module(
            reuse_decision,
            upper_values=(
                most_similar_pre_proj, 
                restored_hidden_states,
                restored_query_states,
                restored_key_states,
                restored_value_states,
            ),
            lower_values=(
                pre_proj,
                hidden_states,
                query_states,
                key_states,
                value_states,
            ),
            hard=hard,
            tau=tau
        )
        
        reuse_map = reuse_map.squeeze(-1)
        if self.skip_cls:
            # Re-attach cls token
            pre_proj = torch.cat((cls_pre_proj, pre_proj), dim=1)
            hidden_states = torch.cat((cls_hidden_states, hidden_states), dim=1)
            query_states = torch.cat((cls_query_states, query_states), dim=1)
            key_states = torch.cat((cls_key_states, key_states), dim=1)
            value_states = torch.cat((cls_value_states, value_states), dim=1)

            reuse_map = torch.cat(
                    (
                        torch.zeros((B, 1), dtype=torch.bool, device=reuse_map.device),
                        reuse_map
                    ),
                    dim=1
                )

        return reuse_map, pre_proj, hidden_states, query_states, key_states, value_states

    def forward_v2(
            self,
            cached_states,
            ref_state,
            *output_states,
            attn_weights=None,
            compressed_map=None,
            ref_mask=None,
            prev_reuse_map=None,
            **kwargs
        ):
        B, N, dim = ref_state.shape


        if self.skip_cls:
            cls_ref_state = ref_state[:, 0:1]
            cls_output_states = [output_state[:, 0:1] for output_state in output_states]

            ref_state = ref_state[:, 1:]
            output_states = [output_state[:, 1:] for output_state in output_states]

            if prev_reuse_map is not None:
                prev_reuse_map = prev_reuse_map[:, 1:]

            lhs_N = N - 1
        else:
            lhs_N = N

        cached_ref_state, *cached_output_states = cached_states

        assert len(cached_output_states) == len(output_states), "Cached and current output states must have the same length"

        similarity = self.similarity_module(ref_state, cached_ref_state)

        if ref_mask is not None:
            similarity = similarity.view(B, lhs_N, -1, N)
            ref_mask = ref_mask.view(B, 1, -1, 1).expand(-1, lhs_N, -1, N)
            similarity[~ref_mask] = -2.
            similarity = similarity.view(B, lhs_N, -1)

        importance = self.importance_module(attn_weights)
        reuse_decision, most_similar_idx, _ = self.decision_module(
            importance=importance,
            similarity=similarity,
            compressed_map=compressed_map,
        )

        # Gather most similar inputs
        most_similar_ref_state = torch.gather(
            cached_ref_state,
            dim=1,
            index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
        ).squeeze(1)

        most_similar_output_states = [
            torch.gather(
                cached_output_state,
                dim=1,
                index=most_similar_idx.unsqueeze(-1).expand(-1, -1, dim),
            ).squeeze(1)
            for cached_output_state in cached_output_states
        ]

        assert isinstance(self.restoration_module, PassthroughRestoration), "Only PassthroughRestoration is supported for forward_v2"

        reuse_map, (
            ref_state,
            *output_states,
        ) = self.gating_module(
            reuse_decision,
            upper_values=(
                most_similar_ref_state,
                *most_similar_output_states,
            ),
            lower_values=(
                ref_state,
                *output_states,
            ),
            hard=kwargs.get('hard', False),
            tau=kwargs.get('tau', None)
        )
        
        reuse_map = reuse_map.squeeze(-1)

        if self.skip_cls:
            # Re-attach cls token
            ref_state = torch.cat((cls_ref_state, ref_state), dim=1)
            output_states = [
                torch.cat((cls_output_state, output_state), dim=1)
                for cls_output_state, output_state in zip(cls_output_states, output_states)
            ]

            reuse_map = torch.cat(
                (
                    torch.zeros((B, 1), dtype=torch.bool, device=reuse_map.device),
                    reuse_map
                ),
                dim=1
            )

        return reuse_map, ref_state, *output_states