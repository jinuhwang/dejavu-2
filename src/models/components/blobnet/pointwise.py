import torch
import torch.nn as nn

class PointWiseTN(nn.Module):
    def __init__(self, channels, drop=0.2, num_output=1):
        super(PointWiseTN, self).__init__()

        layers = []
        for c in channels:
            layers.append(nn.Conv1d(c*num_output, c*num_output, 1, bias=False, groups=num_output))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))

        self.inner = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.num_output = num_output

    def forward(self, x):
        B, FC, T, H, W = x.shape
        residual = x
        # [B, F*C, T, H, W] => [B, F, C, T, H, W]
        x = x.view(B, self.num_output, -1, T, H, W)
        C = x.shape[2]
        # [B, F, C, T, H, W] => [B, C, F, T, H, W]
        x = x.transpose(1, 2)
        # [B, C, F, T, H, W] => [B*C, T*F, H, W]
        x = x.reshape(B*C, -1, H*W)
        x = self.inner(x)
        # [B*C, T*F, H*W] => [B, C, F, T, H, W]
        x = x.view(B, C, self.num_output, T, H, W)
        # [B, C, F, T, H, W] => [B, F, C, T, H, W]
        x = x.transpose(1, 2)
        # [B, F, C, T, H, W] => [B, F*C, T, H, W]
        x = x.reshape(B, -1, T, H, W)
        x = x + residual
        x = self.relu(x)
        return x
