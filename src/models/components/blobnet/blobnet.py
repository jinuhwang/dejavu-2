import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from ..blobnet.preprocessing import Preprocessing  # Assuming this is also migrated to PyTorch

class BlobNet(nn.Module):
    def __init__(
            self,
            input_shape,
            e_spatial_channels,
            e_temporal_channels_list,
            e_spatial_kernel_size,
            e_activation,
            e_use_bn,
            d_spatial_channels,
            d_spatial_kernel_size,
            num_hidden_layers,
            num_outputs=1,
            act='',
            disable_batchnorm=False,
        ):
        super(BlobNet, self).__init__()

        patch_per_side = input_shape[-1]
        
        self.preprocessing = Preprocessing(channels=input_shape[0], disable_batchnorm=disable_batchnorm)
        self.encoder = Encoder(
            spatial_channels=e_spatial_channels,
            temporal_channels_list=e_temporal_channels_list,
            spatial_kernel_size=e_spatial_kernel_size,
            padding="same",
            use_bias=True,
            use_bn=e_use_bn,
            patch_per_side=patch_per_side,
            num_hidden_layers=num_hidden_layers
        )

        input_shape = tuple(input_shape)
        C, T, H, W = input_shape
        dummy_input = torch.zeros(input_shape).unsqueeze(0)
        dummy_input = dummy_input.unsqueeze(1)
        dummy_input = dummy_input.expand(-1, num_hidden_layers, -1, -1, -1, -1)
        dummy_input = dummy_input.reshape(1, C*num_hidden_layers, T, H, W)
        dummy_encoder_output = self.encoder(dummy_input)

        x_rev = [x[:, :, :1] for x in reversed(dummy_encoder_output)]
        shapes = [x.shape for x in x_rev]
        shapes.append(input_shape)

        self.decoder = Decoder(
            input_shapes=shapes,
            spatial_channels=d_spatial_channels,
            spatial_kernel_size=d_spatial_kernel_size,
            patch_per_side=patch_per_side,
            num_hidden_layers=num_hidden_layers,
            num_outputs=num_outputs
        )
        self.input_shape = input_shape
        self.num_hidden_layers = num_hidden_layers

        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == '':
            self.act = nn.Identity()
        else:
            raise ValueError(f"Activation function {act} not supported.")

    def forward(self, x):
        x = self.preprocessing(x)
        B, C, T, H, W = x.shape
        x = x.unsqueeze(1)
        x = x.expand(-1, self.num_hidden_layers, -1, -1, -1, -1)
        x = x.reshape(B, C*self.num_hidden_layers, T, H, W)
        x_enc = self.encoder(x)
        x_rev = [x[:, :, :1] for x in reversed(x_enc)]
        x_dec = self.decoder(x_rev)
        out = x_dec.squeeze(2)
        out = self.act(out)

        return out

# Example usage
if __name__ == '__main__':
    patch_per_side = 16
    model = BlobNet(
        input_shape=[3, 30, patch_per_side, patch_per_side],
        # Encoder
        e_spatial_channels=[(3, 16), (16, 32), (32, 64), (64, 128)],
        e_spatial_kernel_size=[1, 2, 2],
        e_temporal_channels_list=[[30, 30], [30, 30], [30, 30], [30, 30]],
        e_activation='relu',
        e_use_bn=True,
        e_patch_per_side=patch_per_side,
        # Decoder
        d_spatial_channels=[(128, 64), (128, 32), (64, 16), (32, 16)],
        d_spatial_kernel_size=[1, 2, 2],
        num_hidden_layers=12,
    )

    dummy_input = torch.zeros([1, 3, 30, 16, 16])

    dummy_output = model(dummy_input)
    assert dummy_output.shape == (1, 1, 16, 16)

