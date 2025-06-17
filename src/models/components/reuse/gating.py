import torch
from torch import nn
import torch.nn.functional as F

class SteepSigmoid(nn.Module):
    def __init__(self, s=40):
        super().__init__()
        self.s = s

    def forward(self, decision, upper_values, lower_values, **kwargs):
        if decision.shape[-1] == 2:
            decision = decision[..., 0:1] - decision[..., 1:2]
        gate_map = torch.sigmoid(self.s * decision)

        ret = []
        for upper_value, lower_value in zip(upper_values, lower_values):
            upper_term = gate_map * upper_value
            lower_term = (1 - gate_map) * lower_value
            ret.append(upper_term + lower_term)
        return gate_map, ret

def gumbel_softmax(logits, tau=1.0, hard=False):
    """
    Compute the Gumbel-Softmax approximation for discrete distribution sampling.
    :param logits: Logits from which to sample. Shape (batch_size, num_classes)
    :param tau: Temperature parameter. Lower values make the distribution "harder".
    :param hard: If True, returns one-hot encoded result, but gradients are still based on soft approximation.
    :return: Sampled "probabilities" from the Gumbel-Softmax distribution.
    """
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)  # Sample from Gumbel(0, 1)
    y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
    
    if hard:
        # Create a one-hot encoding from the soft sample for the forward pass
        y_hard = torch.argmax(y_soft, dim=-1, keepdim=True)
        y_hard = F.one_hot(y_hard, num_classes=logits.size(-1)).float()
        # Use straight-through estimator for the backward pass
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    
    return y

class GumbelSoftmaxGating(nn.Module):
    def __init__(self, tau=0.25):
        super().__init__()
        self.tau = tau

    def forward(self, decision, upper_values, lower_values, hard=False, tau=None, **kwargs):
        logits = torch.stack((decision, -decision), dim=-1)
        if tau is None:
            tau = self.tau
        choices = gumbel_softmax(logits, tau=tau, hard=hard)
        gate_map = choices[..., 0]

        ret = []
        for upper_value, lower_value in zip(upper_values, lower_values):
            upper_term = choices[..., 0] * upper_value
            lower_term = choices[..., 1] * lower_value
            ret.append(upper_term + lower_term)

        return gate_map, ret


class HardGating(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decision, upper_values, lower_values, **kwargs):
        # gate_map = (decision > 0).float()
        if decision.shape[-1] == 1:
            gate_map = (decision > 0).float()
        elif decision.shape[-1] == 2:
            gate_map = (decision[..., 0] > decision[..., 1]).float().unsqueeze(-1)
        else:
            raise ValueError("HardGating expects a 1-dimensional or 2-dimensional decision tensor")


        ret = []
        for upper_value, lower_value in zip(upper_values, lower_values):
            upper_term = gate_map * upper_value
            lower_term = (1 - gate_map) * lower_value
            ret.append(upper_term + lower_term)

        return gate_map, ret

class AdafuseGating(nn.Module):
    def __init__(self, tau=0.25, gating_scheduling=False):
        super().__init__()
        self.tau = tau
        self.gating_scheduling = gating_scheduling

    def forward(self, decision, upper_values, lower_values, hard=False, tau=None, **kwargs):
        assert decision.shape[-1] == 2, "AdafuseGating expects a 2-dimensional decision tensor"

        logits = torch.log(F.softmax(decision, dim=-1).clamp(min=1e-8))

        # if tau is None:
        if not self.gating_scheduling or tau is None:
            assert tau is None
            tau = self.tau
        
        choices = F.gumbel_softmax(logits, tau=tau, hard=hard)
        gate_map = choices[..., 0]

        ret = []
        for upper_value, lower_value in zip(upper_values, lower_values):
            upper_term = choices[..., 0:1] * upper_value
            lower_term = choices[..., 1:2] * lower_value
            ret.append(upper_term + lower_term)

        return gate_map, ret