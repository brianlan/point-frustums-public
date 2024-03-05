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
        share_weights: bool = True,
        norm_group_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.layers_in = tuple(layers_in)
        self.n_convolutions_classification = n_convolutions_classification
        self.n_convolutions_regression = n_convolutions_regression
        self.share_weights = share_weights
        self.norm_group_size = norm_group_size
        self.dropout = dropout

        self.spec_cls = {"class": n_classes, "attribute": n_attributes}
        self.spec_reg = {"center": 3, "wlh": 3, "orientation": 6, "velocity": 2, "vfl": n_classes}

        self.out_keys = tuple(self.spec_cls.keys()) + tuple(self.spec_reg.keys())
        self.split_sizes_classification = tuple(self.spec_cls.values())
        self.split_sizes_regression = tuple(self.spec_reg.values())

        self.head_classification = self.build_head(self.spec_cls, n_convolutions_classification, share_weights)
        self.head_regression = self.build_head(self.spec_reg, n_convolutions_regression, share_weights)

    def build_head(self, outputs: dict[str, int], n_convolutions: int, share_weights: bool = True) -> nn.ModuleList:
        """
        Builds a head with several convolutional layers and one final layer that stacks the specified outputs.
        If `share_weights` is False, the built head is duplicated, otherwise it is reused for each layer.
        Geometric 2d dropout is applied to the input of each convolution layer and standard dropout right before the
        last, depth-wise convolution.
        :return:
        """
        n_channels_out = sum(outputs.values())

        assert isclose(self.n_channels_in % self.norm_group_size, 0)
        n_normalization_groups = self.n_channels_in // self.norm_group_size

        layers = []
        for _ in range(n_convolutions):
            layers.append(nn.Dropout2d(self.dropout))
            layers.append(Conv2dSpherical(self.n_channels_in, self.n_channels_in, kernel_size=3))
            layers.append(nn.GroupNorm(n_normalization_groups, self.n_channels_in))
            layers.append(nn.GELU())
        layers.append(nn.Dropout(self.dropout))
        layers.append(Conv2dSpherical(self.n_channels_in, n_channels_out, kernel_size=1))
        layers = nn.Sequential(*layers)

        head = nn.ModuleList()
        for _ in self.layers_in:
            if share_weights is False:
                layers = deepcopy(layers)
            head.append(layers)

        return head

    def forward(self, features: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        classification = {}
        regression = {}
        for layer, head_cls, head_reg in zip(self.layers_in, self.head_classification, self.head_regression):
            classification[layer] = head_cls(features[layer])
            regression[layer] = head_reg(features[layer])

        out = {k: {} for k in self.out_keys}
        for layer in self.layers_in:
            cls = classification[layer].split(self.split_sizes_classification, dim=1)
            reg = regression[layer].split(self.split_sizes_regression, dim=1)
            out["class"][layer] = cls[0]
            out["attribute"][layer] = cls[1]
            out["center"][layer] = reg[0]
            out["wlh"][layer] = reg[1]
            out["orientation"][layer] = reg[2]
            out["velocity"][layer] = reg[3]
            out["vfl"][layer] = reg[4]

        return out
