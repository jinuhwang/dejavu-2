import torch

from src.models.components.loss import ReuseLoss


def _make_fake_inputs(B=2, F=3, N=5, D=8, L=2):
    hidden = [torch.randn(B, F, N, D) for _ in range(L)]
    orig_hidden = [h + 0.05 * torch.randn_like(h) for h in hidden]
    output = torch.randn(B, F, D)
    orig_out = output + 0.05 * torch.randn_like(output)
    reuse_maps = torch.rand(B, F, L, N)
    return hidden, output, orig_hidden, orig_out, reuse_maps


def test_reuse_loss_cosine_and_mse_outputs_finite():
    hidden, output, orig_hidden, orig_out, reuse_maps = _make_fake_inputs()

    # Cosine
    loss_fn_cos = ReuseLoss(target_reuse_rate=0.5, hloss_scaler=[0.5, 1.0])
    vals = loss_fn_cos(hidden, output, orig_hidden, orig_out, reuse_maps)
    assert all(torch.isfinite(v if isinstance(v, torch.Tensor) else torch.tensor(v)) for v in vals[:-1])

    # MSE for both hidden and output
    loss_fn_mse = ReuseLoss(
        target_reuse_rate=0.5,
        hloss_scaler=[0.5, 1.0],
        hidden_loss_type='mse',
        output_loss_type='mse',
    )
    vals2 = loss_fn_mse(hidden, output, orig_hidden, orig_out, reuse_maps)
    assert all(torch.isfinite(v if isinstance(v, torch.Tensor) else torch.tensor(v)) for v in vals2[:-1])


def test_reuse_loss_detach_targets_blocks_grad():
    B, F, N, D, L = 1, 2, 4, 6, 2
    hidden = [torch.randn(B, F, N, D, requires_grad=True) for _ in range(L)]
    orig_hidden = [h.detach().clone().requires_grad_(True) for h in hidden]
    output = torch.randn(B, F, D, requires_grad=True)
    orig_out = output.detach().clone().requires_grad_(True)
    reuse_maps = torch.rand(B, F, L, N)

    loss_fn = ReuseLoss(
        target_reuse_rate=0.5,
        hloss_scaler=[1.0, 1.0],
        hidden_loss_type='mse',
        output_loss_type='mse',
        detach_targets=True,
    )
    loss, *_ = loss_fn(hidden, output, orig_hidden, orig_out, reuse_maps)
    loss.backward()

    # No grads should flow to targets when detach_targets=True
    assert orig_out.grad is None
    assert all(t.grad is None for t in orig_hidden)

