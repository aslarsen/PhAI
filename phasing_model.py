import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce


class ConvolutionalBlock(nn.Module):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()

        self.act = nn.GELU()

        self.conv1 = nn.Conv3d(filters, filters, kernel_size = kernel_size, padding = padding)
        self.conv2 = nn.Conv3d(filters, filters, kernel_size = kernel_size, padding = padding)

        self.norm1 = nn.GroupNorm(filters, filters)
        self.norm2 = nn.GroupNorm(filters, filters)

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.act(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.norm2(x)

        x = x + identity
        return x

class MLPLayer(nn.Module):
    def __init__(self, token_nr, dim, dim_exp, mix_type):
        super().__init__()

        self.act    = nn.GELU()

        self.norm1  = nn.GroupNorm(token_nr, token_nr)

        if mix_type == 'token':
            self.layer1 =  nn.Linear(token_nr, dim_exp)
            self.layer2 =  nn.Linear(dim_exp, token_nr)

            self.rearrange1 = Rearrange('b x y -> b y x')
        else:
            self.layer1 =  nn.Linear(dim , dim_exp)
            self.layer2 =  nn.Linear(dim_exp, dim)

            self.rearrange1 = nn.Identity()

        self.mix_type = mix_type

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.rearrange1(x)

        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)

        x = self.rearrange1(x)

        x = x + identity

        return x

class PhAINeuralNetwork(nn.Module):
    def __init__(self, *, max_index, filters, kernel_size, cnn_depth, dim, dim_exp, dim_token_exp, mlp_depth, reflections):
        super().__init__()

        hkl           = [max_index*2+1, max_index+1, max_index+1]
        mlp_token_nr  = filters
        padding       = int((kernel_size - 1) / 2)

        self.net_a = nn.Sequential(
            Rearrange('b x y z  -> b 1 x y z '),

            nn.Conv3d(1, filters, kernel_size = kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters)
        )

        self.net_p = nn.Sequential(
            Rearrange('b x y z  -> b 1 x y z '),

            nn.Conv3d(1, filters, kernel_size = kernel_size, padding=padding),
            nn.GELU(),
            nn.GroupNorm(filters, filters)
        )

        self.net_convolution_layers = nn.Sequential(
            *[nn.Sequential(
                ConvolutionalBlock(filters, kernel_size = kernel_size, padding = padding),
            ) for _ in range(cnn_depth)],
        )

        self.net_projection_layer = nn.Sequential(
            Rearrange('b c x y z  -> b c (x y z)'),
            nn.Linear(hkl[0]*hkl[1]*hkl[2], dim),
        )

        self.net_mixer_layers = nn.Sequential(
            *[nn.Sequential(
                MLPLayer(mlp_token_nr, dim, dim_token_exp, 'token'),
                MLPLayer(mlp_token_nr, dim, dim_exp      , 'channel'),
            ) for _ in range(mlp_depth)],
            nn.LayerNorm(dim),
        )

        self.net_output = nn.Sequential(
            Rearrange('b t x -> b x t'),
            Reduce('b x t -> b x','mean'),

            nn.Linear(dim, reflections*2),
            Rearrange('b (c h) -> b c h ', h = reflections),
        )

    def forward(self, input_amplitudes, input_phases):

        a = self.net_a(input_amplitudes)
        p = self.net_p(input_phases)

        x = a + p

        x = self.net_convolution_layers(x)

        x = self.net_projection_layer(x)

        x = self.net_mixer_layers(x)

        phases = self.net_output(x)

        return phases



