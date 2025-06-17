import torch
from torch import nn

class CLSImportance(nn.Module):
    def __init__(self, exp=2):
        super().__init__()
        self.exp = exp

    def forward(self, attn_weights, **kwargs):
        # [B, H, N, N] => [B, H, N-1]
        cls_attn = attn_weights[:, :, 0, 1:]

        cls_attn = cls_attn ** self.exp

        # [B, H, N-1] => [B, H]
        importance = cls_attn.sum(dim=1)
        return importance

class ZeroTPruenImportance(nn.Module):
    def __init__(self, wpr_iter):
        super().__init__()
        # Thus, to ensure convergence, we set the number # of iterations to 30-50, 5-10, and 1
        #  in the first three layers, medium layers, and last three layers, respectively.
        self.wpr_iter = wpr_iter

        # The results are shown in Fig. 16, which points to the range [0.01,0.7].
        self.var_min = 0.01
        self.var_max = 0.7

    def forward(self, attn_weights, **kwargs):
        # Size of attn_weights is [B, H, N, N]
        B, H, N, _ = attn_weights.shape

        # [B, H, N]
        s = torch.full((B, H, N, 1), 1 / N, device=attn_weights.device)
        # Weighted Parge Rank (WPR)
        # [B, H, N, 1] * [B, N, N, N] = [B, N, N, N]

        # Use the adjacency matrix as a graph shift operator
        a_t = attn_weights.transpose(-1, -2)

        for _ in range(self.wpr_iter):
            s_next = torch.matmul(a_t, s)
            # print(s_next.sum().item(), (s_next - s).abs().max().item())
            s = s_next

        # [B, H, N, 1] => [B, H, N] => [B, H, N-1]
        s = s.squeeze(-1)
        # Variance-based Head Filter (BHF)
        # We compute the variance of the distribution in each head and set both a minimum and a maximum threshold for the variance.
        # Heads with a distribution variance exceeding the maximum threshold or falling below the minimum threshold are excluded from the computation.
        # [B, H, N] => [B, H]

        '''
        head_var = s.var(dim=-1)
        mask = (head_var >= self.var_min) & (head_var <= self.var_max)
        s = s[..., 1:]

        dividend = mask.sum(dim=-1, keepdim=True) + 1e-8
        print(dividend)

        # Emphaisizing Informative Region (EIR) aggregation
        # In order to address this problem, we aggregate the importance scores across heads via a root-mean of sum of squares.
        # ([B, H, N] => [B, N] / [B, H] => [B, 1]) => [B, N]
        importance = torch.sqrt((s**2).sum(dim=-2) * mask / dividend)
        '''
        s = s[..., 1:]
        importance = torch.sqrt((s**2).sum(dim=-2))

        return importance

class NoneImportance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, **kwargs):
        return None

class TldrImportance(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, attn_weights, **kwargs):
        score_attn = attn_weights.mean(dim=1)
        scores = (score_attn * torch.log(score_attn)).sum(dim=2).unsqueeze(-1)

        # BACKGROUND REMOVING
        B, T_R, _ = scores.shape
        scores = scores - scores.amin(dim=1, keepdim=True)
        scores = scores / scores.amax(dim=1, keepdim=True)
        score_mask = scores < scores.mean(dim=1, keepdim=True)

        # FOREGROUND SHARPENING
        scores = scores - scores.mean(dim=1, keepdim=True)
        scores = scores / scores.amax(dim=1, keepdim=True)
        scores[score_mask] = 0.0

        return scores

class MeanImportance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, **kwargs):
        # [B, H, N, N] => [B, _, _, N]
        importance = attn_weights.mean(dim=(1, 2))
    
        return importance