import torch

from src.models.components.reuse.decision import ReuseMLP
from src.models.components.reuse.module import ReuseModule
from src.models.components.reuse.importance import CLSImportance
from src.models.components.reuse.similarity import CosineSimilarity
from src.models.components.reuse.gating import HardGating
from src.models.components.reuse.restoration import PassthroughRestoration


def test_local_ctx_conv_and_neighbors_shapes():
    B, P, D = 2, 16, 64
    tokens = torch.randn(B, P, D)

    mlp_conv = ReuseMLP(layer_pattern='l', local_ctx_mode='conv', local_ctx_rank=8)
    feats_conv = mlp_conv.build_local_ctx(tokens)
    assert feats_conv.shape == (B, P, 3)
    assert torch.isfinite(feats_conv).all()

    # Non-square token count -> falls back to 1xP
    P2 = 10
    tokens2 = torch.randn(B, P2, D)
    mlp_n = ReuseMLP(layer_pattern='l', local_ctx_mode='neighbors', local_ctx_rank=8, local_ctx_kernel=3)
    feats_n = mlp_n.build_local_ctx(tokens2)
    assert feats_n.shape == (B, P2, 3)
    assert torch.isfinite(feats_n).all()


def test_reuse_mlp_forward_accepts_local_ctx():
    B, N, N2 = 2, 7, 9
    importance = torch.rand(B, N - 1)
    similarity = torch.rand(B, N - 1, N2)
    compressed_map = torch.rand(B, N - 1)

    mlp = ReuseMLP(layer_pattern='l', out_dim=2, local_ctx_mode='conv')
    local_ctx = torch.randn(B, N - 1, 3)
    decision, idx, _ = mlp(importance, similarity, compressed_map, local_ctx=local_ctx)
    assert decision.shape == (B, N - 1, 2)
    assert idx.shape == (B, N - 1)


def test_reuse_module_builds_and_passes_local_ctx():
    # Minimal integration: ensure local_ctx path runs and shapes are valid
    B, N, D = 1, 5, 8  # includes CLS
    F = 2              # cached frames
    Hh = 1             # attn heads

    decision = ReuseMLP(layer_pattern='l', out_dim=2, local_ctx_mode='conv', local_ctx_rank=4)
    from src.models.components.reuse.similarity import LocalCosineSimilarity
    similarity = LocalCosineSimilarity()
    importance = CLSImportance()
    gating = HardGating()
    restoration = PassthroughRestoration()
    module = ReuseModule(decision, similarity, importance, gating, restoration, skip_cls=True)

    pre_proj = torch.randn(B, N, D)
    hidden_states = torch.randn(B, N, D)
    query_states = torch.randn(B, N, D)
    key_states = torch.randn(B, N, D)
    value_states = torch.randn(B, N, D)

    cached_pre_proj = torch.randn(B, F * N, D)
    cached_hidden_states = torch.randn(B, F * N, D)
    cached_query_states = torch.randn(B, F * N, D)
    cached_key_states = torch.randn(B, F * N, D)
    cached_value_states = torch.randn(B, F * N, D)

    attn_weights = torch.rand(B, Hh, N, N)
    ref_mask = torch.ones(B, F, dtype=torch.bool)

    reuse_map, *outs = module(
        cached_states=(
            cached_pre_proj,
            cached_hidden_states,
            cached_query_states,
            cached_key_states,
            cached_value_states,
        ),
        pre_proj=pre_proj,
        hidden_states=hidden_states,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attn_weights=attn_weights,
        ref_mask=ref_mask,
    )
    assert reuse_map.shape == (B, N)
    for t in outs:
        assert t.shape == (B, N, D)
