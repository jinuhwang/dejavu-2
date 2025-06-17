import torch
import torch.nn as nn

class Preprocessing(nn.Module):
    def __init__(self, channels=3, disable_batchnorm=False):
        super(Preprocessing, self).__init__()
        # If there are additional layers or operations, initialize them here
        if disable_batchnorm:
            self.bn = nn.Identity()
        else:
            self.bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        x = torch.abs(x)

        mb_type = x[:, :1]
        mvs = x[:, 1:3]
        qp = x[:, 3:]
        mb_type = torch.clamp(mb_type, 0.0, 6.0) / 6.0
        qp = qp / 51.0

        x = torch.cat([mb_type, mvs, qp], dim=1)

        x = self.bn(x)

        return x

# Example usage
if __name__ == "__main__":
    # Example input tensor for ViT-L/14
    inp = torch.rand(4, 3, 4, 16, 16)  
    print(Preprocessing()(inp).shape)
