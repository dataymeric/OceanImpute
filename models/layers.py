import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(self, in_features, hidden_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        bias=True,
        kernels_per_layer=1,
    ):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels * kernels_per_layer, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        bias=True,
        kernels_per_layer=1,
    ):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv3d(
            in_channels * kernels_per_layer, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Conv2Plus1d(nn.Module):
    """Conv(2+1)d layer. https://arxiv.org/pdf/1711.11248v3"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        bias=True,
    ):
        super(Conv2Plus1d, self).__init__()
        t = kernel_size[0]
        d = (kernel_size[1] + kernel_size[2]) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = int(
            (t * d**2 * in_channels * out_channels)
            / (d**2 * in_channels + t * out_channels)
        )

        self.conv2d = nn.Conv2d(
            in_channels,
            self.hidden_size,
            kernel_size=(kernel_size[1], kernel_size[2]),
            stride=(stride[1], stride[2]),
            padding=(padding[1], padding[2]) if "same" not in padding else "same",
            bias=bias,
        )
        self.conv1d = nn.Conv1d(
            self.hidden_size,
            out_channels,
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0] if "same" not in padding else "same",
            bias=bias,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, t, h, w = x.size()

        x = x.view(b * t, c, h, w)
        x = self.conv2d(x)
        x = self.relu(x)  # (b * t, hidden_size, h, w)

        h, w = x.size(2), x.size(3)
        x = x.view(-1, self.hidden_size, t)  # (b * h * w, hidden_size, t)
        x = self.conv1d(x)  # (b * h * w, out_channels, t)

        t = x.size(2)
        x = x.view(b, self.out_channels, t, h, w)  # (b, out_channels, t, h, w)
        return x


class PartialConv2d(nn.Conv2d):
    """Partial Convolution 2d Layer. https://arxiv.org/pdf/1804.07723

    https://github.com/NVIDIA/partialconv
    """

    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_mask_updater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            self.weight_mask_updater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.slide_winsize = (
            self.weight_mask_updater.shape[1]
            * self.weight_mask_updater.shape[2]
            * self.weight_mask_updater.shape[3]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_mask_updater.type() != input.type():
                    self.weight_mask_updater = self.weight_mask_updater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_mask_updater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConv3d(nn.Conv3d):
    """Partial Convolution 3d Layer. https://arxiv.org/pdf/1804.07723

    https://github.com/NVIDIA/partialconv
    """

    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv3d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_mask_updater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
            )
        else:
            self.weight_mask_updater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
            )

        self.slide_winsize = (
            self.weight_mask_updater.shape[1]
            * self.weight_mask_updater.shape[2]
            * self.weight_mask_updater.shape[3]
            * self.weight_mask_updater.shape[4]
        )

        self.last_size = (None, None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 5
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_mask_updater.type() != input.type():
                    self.weight_mask_updater = self.weight_mask_updater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                            input.data.shape[4],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1,
                            1,
                            input.data.shape[2],
                            input.data.shape[3],
                            input.data.shape[4],
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv3d(
                    mask,
                    self.weight_mask_updater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask) / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv3d, self).forward(
            torch.mul(input, mask_in) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer, layer_norm):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv_layer(
            in_channels, out_channels, kernel_size=3, padding="same"
        )
        self.norm1 = layer_norm(out_channels)
        self.conv2 = conv_layer(
            out_channels, out_channels, kernel_size=3, padding="same"
        )
        self.norm2 = layer_norm(out_channels)

        if in_channels != out_channels:
            self.shortcut = conv_layer(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + self.shortcut(x)
        out = self.relu(out)

        return out
