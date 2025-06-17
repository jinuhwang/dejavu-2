import torch
import torch.nn as nn
from .pointwise import PointWiseTN  # Assuming PointWiseTN is already migrated to PyTorch

class Encoder(nn.Module):

    def __init__(
            self,
            spatial_channels,
            temporal_channels_list,
            spatial_kernel_size,
            padding='same',
            drop=0.1,
            activation='relu',
            use_bias=True,
            use_bn=True,
            patch_per_side=None,
            num_hidden_layers=1,
        ):
        super(Encoder, self).__init__()

        assert len(spatial_kernel_size) == 3
        assert len(spatial_channels) == len(temporal_channels_list)

        self.contracting = nn.ModuleList()
        self.num_output = num_hidden_layers

        for s_channels, t_channels in zip(spatial_channels, temporal_channels_list):
            seq = nn.Sequential()
            sc_in, sc_out = s_channels

            sc_in = sc_in * num_hidden_layers
            sc_out = sc_out * num_hidden_layers

            seq.append(nn.Conv3d(sc_in, sc_out, spatial_kernel_size, padding=padding, bias=use_bias, groups=num_hidden_layers))
            seq.append(nn.ReLU())

            if patch_per_side % 2:
                patch_per_side += 1
                seq.append(nn.ZeroPad2d((0,1,0,1)))
            else:
                seq.append(nn.Identity())
            patch_per_side /= 2

            if use_bn:
                seq.append(nn.GroupNorm(num_hidden_layers, sc_out))
            else:
                seq.append(nn.Identity())

            seq.append(nn.MaxPool3d((1,2,2)))
            
            seq.append(PointWiseTN(channels=t_channels, num_output=num_hidden_layers))

            self.contracting.append(seq)

    def forward(self, x):
        ret = ()
        for layer in self.contracting:
            x = layer(x)
            ret += x,
        return ret
