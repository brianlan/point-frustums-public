from collections.abc import Mapping
from copy import deepcopy
from math import isclose

from torch import nn, Tensor

from point_frustums.ops.spherical_coos_convolutions import Conv2dSpherical
from ..base_models import Head


class PointFrustumsHead(Head):
    def __init__(
        self,
        n_channels_in: int,
        layers_in: list[str],
        n_classes: int,
        n_attributes: int,
        n_convolutions_classification: int = 4,
        n_convolutions_regression: int = 4,
        share_weights_conv: bool = True,
        share_weights_heads: bool = True,
        norm_group_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.layers_in = layers_in
        self.n_classes = n_classes
        self.n_attributes = n_attributes
        self.n_convolutions_classification = n_convolutions_classification
        self.n_convolutions_regression = n_convolutions_regression
        self.share_weights_conv = share_weights_conv
        self.share_weights_heads = share_weights_heads
        self.norm_group_size = norm_group_size
        self.dropout = dropout

        self.conv_classification = nn.ModuleDict()
        self.head_class = nn.ModuleDict()
        self.head_attribute = nn.ModuleDict()

        self.conv_regression = nn.ModuleDict()
        self.head_center = nn.ModuleDict()
        self.head_wlh = nn.ModuleDict()
        self.head_orientation = nn.ModuleDict()
        self.head_velocity = nn.ModuleDict()
        self.head_iou = nn.ModuleDict()

        self.build_layers()

    def build_layers(self):
        """Build the layers required for the detection head.
        First, build the zeroth layer and then either reference the latter or create a copy for the remaining layers.

        Geometric 2d dropout is applied to the input of each convolution layer and standard dropout right before heads.

        Weights can either be shared or decoupled. This is configurable for the convolutional layers and the heads.
        The aim should be to map each layer to a shared vector space and then continue with shared weights. Given the
        post layer of the FPN achieves this, then we can share all weights of the detection head.
        :return:
        """
        zero_layer = self.layers_in[0]
        assert isclose(self.n_channels_in % self.norm_group_size, 0)
        n_normalization_groups = self.n_channels_in // self.norm_group_size

        def build_conv(n_layers):
            conv = []
            for _ in range(n_layers):
                conv.append(nn.Dropout2d(self.dropout))
                conv.append(Conv2dSpherical(self.n_channels_in, self.n_channels_in, kernel_size=3))
                conv.append(nn.GroupNorm(n_normalization_groups, self.n_channels_in))
                conv.append(nn.GELU())
            conv.append(nn.Dropout(self.dropout))
            return nn.Sequential(*conv)

        self.conv_classification[zero_layer] = build_conv(self.n_convolutions_classification)
        self.head_class[zero_layer] = Conv2dSpherical(self.n_channels_in, self.n_classes, kernel_size=1)
        self.head_attribute[zero_layer] = Conv2dSpherical(self.n_channels_in, self.n_attributes, kernel_size=1)

        self.conv_regression[zero_layer] = build_conv(self.n_convolutions_regression)
        self.head_center[zero_layer] = Conv2dSpherical(self.n_channels_in, 3, kernel_size=1)
        self.head_wlh[zero_layer] = Conv2dSpherical(self.n_channels_in, 3, kernel_size=1)
        self.head_orientation[zero_layer] = Conv2dSpherical(self.n_channels_in, 6, kernel_size=1)
        self.head_velocity[zero_layer] = Conv2dSpherical(self.n_channels_in, 2, kernel_size=1)
        self.head_iou[zero_layer] = Conv2dSpherical(self.n_channels_in, self.n_classes, kernel_size=1)

        def identity(module: nn.Module) -> nn.Module:
            return module

        propagate_function_conv = identity if self.share_weights_conv else deepcopy
        propagate_function_head = identity if self.share_weights_heads else deepcopy

        for layer in self.layers_in[1:]:
            self.conv_classification[layer] = propagate_function_conv(self.conv_classification[zero_layer])
            self.head_class[layer] = propagate_function_head(self.head_class[zero_layer])
            self.head_attribute[layer] = propagate_function_head(self.head_attribute[zero_layer])
            self.conv_regression[layer] = propagate_function_conv(self.conv_regression[zero_layer])
            self.head_center[layer] = propagate_function_head(self.head_center[zero_layer])
            self.head_wlh[layer] = propagate_function_head(self.head_wlh[zero_layer])
            self.head_orientation[layer] = propagate_function_head(self.head_orientation[zero_layer])
            self.head_velocity[layer] = propagate_function_head(self.head_velocity[zero_layer])
            self.head_iou[layer] = propagate_function_head(self.head_iou[zero_layer])

    def forward(self, features: Mapping[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        output = {"class": {}, "attribute": {}, "center": {}, "wlh": {}, "orientation": {}, "velocity": {}, "iou": {}}
        for layer in self.layers_in:
            classification = self.conv_classification[layer](features[layer])
            output["class"][layer] = self.head_class[layer](classification)
            output["attribute"][layer] = self.head_attribute[layer](classification)

            regression = self.conv_regression[layer](features[layer])
            output["center"][layer] = self.head_center[layer](regression)
            output["wlh"][layer] = self.head_wlh[layer](regression)
            output["orientation"][layer] = self.head_orientation[layer](regression)
            output["velocity"][layer] = self.head_velocity[layer](regression)
            output["iou"][layer] = self.head_iou[layer](regression)
        return output
