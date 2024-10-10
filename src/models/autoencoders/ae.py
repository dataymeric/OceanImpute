import torch.nn as nn


class Conv3dAutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, num_blocks=3):
        super(Conv3dAutoEncoder, self).__init__()

        self.num_blocks = num_blocks
        self.embed_dim = embed_dim

        # Calculate the number of filters for each block
        filter_sizes = [in_channels * (2**i) for i in range(num_blocks)]

        # Encoder
        encoder_layers = []
        for i in range(num_blocks):
            encoder_layers.append(
                nn.Conv3d(
                    filter_sizes[i],
                    filter_sizes[i + 1] if i < num_blocks - 1 else embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(num_blocks):
            decoder_layers.append(
                nn.ConvTranspose3d(
                    embed_dim if i == 0 else filter_sizes[num_blocks - i],
                    filter_sizes[num_blocks - i - 1]
                    if i < num_blocks - 1
                    else out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            decoder_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
