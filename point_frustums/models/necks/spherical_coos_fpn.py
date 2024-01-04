from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, conv1x1

from point_frustums.ops.spherical_coos_convolutions import Conv2dSpherical


@dataclass(slots=True, kw_only=True)
class ConfigFPNLayer:
    n_blocks: int
    n_channels: int
    downsampling_horizontal: bool
    downsampling_vertical: bool

    @property
    def stride(self):
        stride_horizontal, stride_vertical = 1, 1
        if self.downsampling_horizontal:
            stride_horizontal = 2
        if self.downsampling_vertical:
            stride_vertical = 2
        return stride_horizontal, stride_vertical


class BottleneckRangeImage(Bottleneck):
    """
    Modifies the standard Bottleneck block to use range image padding with the 3x3 convolution
    """

    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        **kwargs
    ):
        """
        Overwrite the conv2 attribute to a custom convolution with circular padding along the azimuthal (horizontal)
        dimension and zero-padding along the polar (vertical) dimension.
        :param planes:
        :param stride:
        :param groups:
        :param base_width:
        :param dilation:
        :param kwargs:
        """
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            **kwargs
        )
        width = int(planes * (base_width / 64.0)) * groups
        self.conv2 = Conv2dSpherical(
            in_channels=width,
            out_channels=width,
            kernel_size=3,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias=False,
        )


class FPN(nn.Module):
    block = BottleneckRangeImage
    norm_layer = nn.BatchNorm2d

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        layers: Mapping[str, ConfigFPNLayer],
        with_p6: bool = False,
        with_p7: bool = False,
        dropout: float = 0.0,
        upsampling_mode: str = "bilinear",
    ):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.dropout = dropout
        self.upsampling_mode = upsampling_mode
        self.layers = dict(sorted(layers.items()))

        self.strides = {}
        self.up, self.lateral, self.down, self.post = self.build_layers()

        self.p6, self.p7 = None, None
        if with_p6:
            self.p6 = nn.Sequential(
                nn.Dropout(self.dropout),
                Conv2dSpherical(self.n_channels_out, self.n_channels_out, kernel_size=3, stride=2),
                self.norm_layer(self.n_channels_out),
            )
            self.strides["p6"] = tuple(2 * s for s in self.strides[max(self.layers.keys())])
        if with_p7:
            assert with_p6
            self.p7 = nn.Sequential(
                nn.Dropout(self.dropout),
                Conv2dSpherical(self.n_channels_out, self.n_channels_out, kernel_size=3, stride=2),
                self.norm_layer(self.n_channels_out),
            )
            self.strides["p7"] = tuple(2 * s for s in 2 * self.strides["p6"])

    def _get_first_block(self, n_channels_in, n_channels, n_channels_out, stride):
        """
        The first block of each layer optionally performs downsampling.
        - If the layer is configured with a stride > 1
        - Or if the number of input channels is not equal to the number of output channels (expanded working planes)
        """
        downsample = None
        if stride != 1 or n_channels_in != n_channels_out:
            downsample = nn.Sequential(
                conv1x1(n_channels_in, n_channels_out, stride),
                self.norm_layer(n_channels_out),
            )
        block = self.block(
            inplanes=n_channels_in,
            planes=n_channels,
            stride=stride,
            downsample=downsample,
            norm_layer=self.norm_layer,
        )
        return block

    def build_layers(self) -> tuple[torch.ModuleDict, torch.ModuleDict, torch.ModuleDict, torch.ModuleDict]:
        """Build the layers of the FPN.
        A FPN layer consists of several components:
            1. The top-down path is made up of a series of blocks (the first sets the size and dimension for the layer)
            2. The bottom-up path is made up of a simple upsampling layer
            3. The lateral connection maps the output of the bottom-up to the output dimension
            4. The post step applies a 3x3 convolution to the sum of (bottom-up[i] + top_down[i]) to mitigate artifacts
        and the procedure is:
        1. Evaluate entire bottom-up path and the lateral connection and store each output in the results dictionary
        2. Propagate from the highest level down to evaluate the top-down path, at each layer add to the results and
           apply the post-step

        Dropout is applied throughout the network:
        - Geometric 2d dropout before every block in the bottom-up path (and before the conv layers of p6/p6)
        - Standard dropout is applied in the beginning of the lateral connection
        - Geometric 2d dropout before the post layer
        :return:
        """
        n_channels_in = self.n_channels_in
        stride = (1, 1)

        up, lateral, down, post = {}, {}, {}, {}
        previous_layer = None
        for name, layer in self.layers.items():
            stride_layer = layer.stride
            stride = tuple(s_l * s for s_l, s in zip(stride_layer, stride))
            n_channels_out = layer.n_channels * self.block.expansion
            self.strides[name] = stride
            # Step 1: The first block of each layer is a special case as it optionally performs the downsampling
            layers = [
                nn.Dropout2d(self.dropout),
                self._get_first_block(n_channels_in, layer.n_channels, n_channels_out, stride_layer),
            ]
            # Step 2: Add the remaining blocks (no stride/downsampling)
            for _ in range(layer.n_blocks):
                layers.append(nn.Dropout2d(self.dropout))
                layers.append(self.block(n_channels_out, layer.n_channels, norm_layer=self.norm_layer))
            up[name] = nn.Sequential(*layers)
            # Step 3: Build the lateral connection (Conv2D) and the top-down-path upsampling layer (interpolate)
            lateral[name] = nn.Sequential(nn.Dropout(self.dropout), conv1x1(n_channels_out, self.n_channels_out))
            n_channels_in = n_channels_out

            if previous_layer is not None:
                # Delay initialization of the bottom-up pathway by one layer as it is dependent on the specification of
                # the consecutive layer
                down[previous_layer] = nn.Upsample(
                    scale_factor=stride_layer, mode=self.upsampling_mode, align_corners=True
                )
                post[previous_layer] = nn.Sequential(
                    nn.Dropout2d(self.dropout),
                    Conv2dSpherical(self.n_channels_out, self.n_channels_out, kernel_size=3),
                    self.norm_layer(self.n_channels_out),
                )
            previous_layer = name

        return nn.ModuleDict(up), nn.ModuleDict(lateral), nn.ModuleDict(down), nn.ModuleDict(post)

    def forward(self, input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the FPN"""
        results = {}

        # Walk through the bottom-up path and directly apply the lateral FC layer
        for layer in self.layers.keys():
            input = self.up[layer](input)
            results[layer] = self.lateral[layer](input)

        # Walk through the top-down path, add the up-sampled featuremaps to the intermediate result, apply the FC layer
        input = None
        for layer in reversed(self.layers.keys()):
            if input is None:
                # The upmost layer is used directly as returned by the bottom-up path
                input = results[layer]
                continue
            results[layer] += self.down[layer](input)
            input = results[layer]
            results[layer] = self.post[layer](results[layer])

        if self.p6 is not None:
            results["p6"] = self.p6(results[max(self.layers.keys())])
        if self.p7 is not None:
            results["p7"] = self.p6(results["p6"])

        return results
