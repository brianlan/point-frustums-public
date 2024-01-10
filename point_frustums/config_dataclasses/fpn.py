from dataclasses import dataclass


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
        return stride_vertical, stride_horizontal


@dataclass(slots=True, kw_only=True)
class ConfigFPNExtraLayer:
    downsampling_horizontal: bool
    downsampling_vertical: bool

    @property
    def stride(self):
        stride_horizontal, stride_vertical = 1, 1
        if self.downsampling_horizontal:
            stride_horizontal = 2
        if self.downsampling_vertical:
            stride_vertical = 2
        return stride_vertical, stride_horizontal
