import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            desired_shape,
            use_bias=True,
            use_bn=True,
            use_dropout=False,
            use_relu=False,
            num_output=1
        ):
        super(UpsampleBlock, self).__init__()

        self.desired_shape = desired_shape
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=(1, 2, 2), padding=0, bias=use_bias, groups=num_output)
        self.use_dropout = use_dropout
        self.use_relu = use_relu

        self.pre = nn.Sequential()
        if self.use_relu:
            self.pre.append(nn.ReLU())
        if self.use_dropout:
            self.pre.append(nn.Dropout(p=.2))
        


    def forward(self, x):
        x = self.pre(x)
        x = self.conv(x)

        # s = x.shape

        # Crop if necessary
        # if s[-2] > self.desired_shape[-2]:
        #     x = x[..., :self.desired_shape[-2], :]

        # if s[-1] > self.desired_shape[-1]:
        #     x = x[..., :self.desired_shape[-1]]

        return x

class Decoder(nn.Module):
    def __init__(
            self,
            input_shapes,
            spatial_channels,
            spatial_kernel_size,
            use_relu=True,
            use_dropout=True,
            patch_per_side=14,
            num_hidden_layers=12,
            num_outputs=1,
        ):
        super(Decoder, self).__init__()

        assert len(input_shapes) == len(spatial_channels) + 1

        # 2, 4, 7 for input 14
        self.patch_per_sides = []
        for i in range(len(spatial_channels)-1):
            patch_per_side = (patch_per_side+1) // 2
            self.patch_per_sides.append(patch_per_side)
        
        self.patch_per_sides = list(reversed(self.patch_per_sides))

        self.expanding = nn.ModuleList()

        for i, spatial_channel in enumerate(spatial_channels):
            seq = nn.Sequential()
            sc_in, sc_out = spatial_channel
            sc_in = sc_in * num_hidden_layers
            sc_out = sc_out * num_hidden_layers
            seq.append(UpsampleBlock(
                sc_in,
                sc_out,
                spatial_kernel_size,
                input_shapes[i + 1],
                use_relu=use_relu,
                use_dropout=use_dropout,
                num_output=num_hidden_layers
            ))
            if i < len(spatial_channels) - 1:
                seq.append(nn.GroupNorm(num_hidden_layers, sc_out))
            self.expanding.append(seq)

        self.final = nn.Conv3d(sc_out, num_hidden_layers * num_outputs, 1, groups=num_hidden_layers)

    def forward(self, inputs):
        x = inputs[0]
        for i, seq in enumerate(self.expanding[:-1]):
            x = seq(x)
            crop = self.patch_per_sides[i]
            x = torch.cat([x[..., :crop, :crop], inputs[i+1]], dim=1)

        x = self.expanding[-1][0](x)
        out = self.final(x)

        return out
