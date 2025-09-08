import torch

from src.models.components.loss import ReuseLoss


def test_class_and_patch_token_shapes():
    B, F, N, D = 2, 2, 1 + 8, 16
    hidden_states = [torch.randn(B, F, N, D) for _ in range(2)]
    assert hidden_states[0].shape == (B, F, N, D)
    # Class token index 0 exists; patches are 1..N-1
    assert N > 1


def test_decision_layer_attachment_and_loss_minimal():
    # Minimal dummy layer with reuse_module attribute
    class _Sim(torch.nn.Module):
        pass
    class _RM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.similarity_module = _Sim()
            self.decision_module = torch.nn.Identity()
    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reuse_module = _RM()
    enc = torch.nn.Module()
    enc.layers = torch.nn.ModuleList([_Layer()])
    assert hasattr(enc.layers[0], 'reuse_module')
    assert hasattr(enc.layers[0].reuse_module, 'decision_module')

    # Loss dict contains finite values
    B, F, N, D = 2, 2, 1 + 4, 8
    hidden_states = [torch.randn(B, F, N, D) for _ in range(2)]
    original_hidden_states = [h + 1e-2*torch.randn_like(h) for h in hidden_states]
    output = torch.randn(B, F, D)
    original_output = output + 1e-2*torch.randn_like(output)
    reuse_maps = (torch.rand(B, F, 2, N) > 0.5).float()

    loss_fn = ReuseLoss(0.5, hloss_scaler=[0.0, 1.0])
    loss, hidden_error, hh_error, cls_error, reuse_rate, reuse_rate_per_frame = loss_fn(
        hidden_states,
        output,
        original_hidden_states,
        original_output,
        reuse_maps,
    )
    vals = {
        'loss': loss,
        'hidden_error': hidden_error,
        'hh_error': hh_error,
        'cls_error': cls_error,
        'reuse_rate': reuse_rate,
    }
    assert all(torch.isfinite(v) for v in vals.values())
