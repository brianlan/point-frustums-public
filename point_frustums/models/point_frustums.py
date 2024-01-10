from math import prod, exp, isclose

import torch

from point_frustums.config_dataclasses.dataset import DatasetConfig
from point_frustums.config_dataclasses.point_frustums import ModelOutputSpecification
from point_frustums.utils.targets import Targets
from .backbones import PointFrustumsBackbone
from .base_models import Detection3DModel
from .base_runtime import Detection3DRuntime
from .heads import PointFrustumsHead
from .necks import PointFrustumsNeck


class PointFrustumsModel(Detection3DModel):
    def __init__(self, backbone: PointFrustumsBackbone, neck: PointFrustumsNeck, head: PointFrustumsHead):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head


class PointFrustums(Detection3DRuntime):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        *args,
        model: PointFrustumsModel,
        dataset: DatasetConfig,
        # losses: LossesConfig,
        **kwargs,
    ):
        super().__init__(*args, model=model, dataset=dataset, **kwargs)
        # TODO: Add loss functions
        # TODO: Add postprocessing step
        self.featuremap_parametrization = self.register_featuremap_parametrization()

    def evaluate_receptive_fields(self) -> tuple[dict[str, float], dict[str, float]]:
        def get_rf(k: list, s: list) -> int:
            """
            https://distill.pub/2019/computing-receptive-fields/#other-network-operations
            Araujo, et al., "Computing Receptive Fields of Convolutional Neural Networks", Distill, 2019.
            :param k:
            :param s:
            :return:
            """
            n_layers = len(k)
            assert n_layers == len(s)

            r = 0
            for n_layers in range(0, n_layers):
                r += (k[n_layers] - 1) * prod(s[0:n_layers])
            return int(r + 1)

        def get_erf(rf: int) -> float:
            """
            arXiv:1701.04128
            :param rf:
            :return:
            """
            return rf * exp(-0.43)

        layers = []

        kernels = {}
        strides_pol = {}
        strides_azi = {}

        _kernels_prev = []
        _strides_pol_prev = []
        _strides_azi_prev = []

        for layer, specs in self.model.neck.fpn.layers.items():
            layers.append(layer)
            kernels[layer] = _kernels_prev + specs.n_blocks * [3]
            layer_stride_pol, layer_stride_azi = specs.stride
            strides_pol[layer] = _strides_pol_prev + [layer_stride_pol] + (specs.n_blocks - 1) * [1]
            strides_azi[layer] = _strides_azi_prev + [layer_stride_azi] + (specs.n_blocks - 1) * [1]

            _kernels_prev = kernels[layer]
            _strides_pol_prev = strides_pol[layer]
            _strides_azi_prev = strides_azi[layer]

        for extra_layer, specs in self.model.neck.fpn.extra_layers.items():
            layers.append(extra_layer)
            kernels[extra_layer] = _kernels_prev + [3]
            strides_pol[extra_layer] = _strides_pol_prev + [specs.stride[0]]
            strides_azi[extra_layer] = _strides_azi_prev + [specs.stride[1]]

            _kernels_prev = kernels[extra_layer]
            _strides_pol_prev = strides_pol[extra_layer]
            _strides_azi_prev = strides_azi[extra_layer]

        n = min(self.model.head.n_convolutions_classification, self.model.head.n_convolutions_regression)
        for layer in layers:
            kernels[layer].extend(n * [3])
            strides_pol[layer].extend(n * [1])
            strides_azi[layer].extend(n * [1])

        receptive_fields_pol = {l: get_erf(get_rf(kernels[l], strides_pol[l])) for l in layers}
        receptive_fields_azi = {l: get_erf(get_rf(kernels[l], strides_azi[l])) for l in layers}

        return receptive_fields_pol, receptive_fields_azi

    def register_featuremap_parametrization(self) -> ModelOutputSpecification:
        """
        Register the parameter-set that describe the geometric relation between feature vectors and spherical
        coordinates. They are used to map targets to the feature map and predictions back.
        Those parameters need not be optimized but are required (on the correct device) to run the model and shall
        therefore be registered as buffers.

        Torch does not currently provide adequate methods to do so.
        Requirement specification:
        - Parameters need to be managed automatically together with the model (device, fp precision, etc.)
        - Multi-layer featuremaps require a sequence of parameters (one for each layer)
        - Parameters should possibly be grouped semantically
        Possible approaches:
        - Combine parameters into dedicated module -> But: the purpose of a Module is a different one
        - Use a NestedTensor -> But: NestedTensor is still in prototype stage
        - Use BufferDict -> But: not yet available
        Temporary solution:
        - Iteratively register all parameters on the top level and separately
        - Keep a reference to each parameter in a regular dict

        :return:
        """
        layers_rfs_pol, layers_rfs_azi = self.evaluate_receptive_fields()
        layers_rfs = list(zip(layers_rfs_pol.values(), layers_rfs_azi.values()))
        reference_boxes_centered = 0.5 * torch.tensor(layers_rfs).repeat(1, 2) * torch.tensor([-1, -1, 1, 1])

        discrete = self.model.backbone.lidar.discretize
        strides = self.model.neck.fpn.strides.values()

        size_per_layer = []
        size_per_layer_flat = []
        feat_center_grid = []
        for layer, stride in self.model.neck.fpn.strides.items():
            assert isclose(discrete.n_splits_pol % stride[0], 0), (
                f"The polar input featuremap resolution {discrete.n_splits_pol} is not divisible by the stride "
                f"{stride[0]} on layer {layer}."
            )
            assert isclose(discrete.n_splits_azi % stride[1], 0), (
                f"The azimuthal input featuremap resolution {discrete.n_splits_azi} is not divisible by the "
                f"stride {stride[1]} on layer {layer}."
            )
            size_per_layer.append((int(discrete.n_splits_pol / stride[0]), int(discrete.n_splits_azi / stride[1])))
            size_per_layer_flat.append(int(discrete.n_splits / prod(stride)))

            # Get the node position w.r.t. the original input data shape
            nodes_pol = torch.arange(0, discrete.n_splits_pol - 1e-8, step=stride[0])
            nodes_azi = torch.arange(0, discrete.n_splits_azi - 1e-8, step=stride[1])

            # Create meshgrid representation for easy indexing
            nodes_pol, nodes_azi = torch.meshgrid(nodes_pol, nodes_azi, indexing="xy")
            nodes_pol, nodes_azi = nodes_pol.flatten(), nodes_azi.flatten()
            # Merge to obtain the [x, y] coordinates of the centers
            feat_center_grid.append(torch.stack((nodes_azi, nodes_pol), dim=1))
            # size_per_layer.append(feat_center_grid[-1].shape[0])
        feat_center_grid = torch.cat(feat_center_grid, dim=0)
        self.register_buffer("feat_center_grid", feat_center_grid)

        feat_receptive_fields = reference_boxes_centered.repeat_interleave(torch.tensor(size_per_layer_flat), dim=0)
        feat_receptive_fields += feat_center_grid.repeat(1, 2)
        self.register_buffer("feat_receptive_fields", feat_receptive_fields)

        feat_receptive_field_sizes = torch.tensor(layers_rfs).repeat_interleave(
            torch.tensor(size_per_layer_flat), dim=0
        )
        self.register_buffer("feat_receptive_field_sizes", feat_receptive_field_sizes)

        # Scale-scale-shift grid centers to angular center coordinates
        feat_center_pol_azi = feat_center_grid.clone()
        feat_center_pol_azi.div_(torch.tensor([discrete.n_splits_pol, discrete.n_splits_azi]))
        feat_center_pol_azi.mul_(torch.tensor([discrete.range_pol, discrete.range_azi]))
        feat_center_pol_azi.add_(torch.tensor([discrete.fov_pol[0], discrete.fov_azi[0]]))
        self.register_buffer("feat_center_pol_azi", feat_center_pol_azi)

        return ModelOutputSpecification(
            strides=list(strides),
            layer_sizes=size_per_layer,
            layer_sizes_flat=size_per_layer_flat,
        )

    def get_loss(self, predictions: torch.Tensor, targets: Targets):
        # TODO: Match predictions to targets (OTA)
        #   - Requires:
        #       - Predictions
        #       - Targets
        #       - Targets projections
        #       - Featuremap Geometry Specifications
        # TODO: Evaluate foreground index

        # TODO: Classification Loss (Focal Loss)
        # TODO: Attribute Loss (Focal Loss)
        # TODO: Vario-Focal Loss (applied to class-wise IoU Prediction)
        #   - detach box center/wlh/orientation prediction

        # TODO: Subset predictions and broadcast targets (to foreground indices)

        # TODO: 3D Center Loss (Smooth L1 Loss; evaluate only if targets)
        #   - Applied separately to radius and angular components
        #   - Operate on featuremap position invariant targets
        #   - Scale angular component with discretization
        #   - Requires:
        #       -> Angular center of feature vectors (depend on: discretization + strides)
        #       -> Foreground
        # TODO: WLH Loss (Smooth L1 Loss; evaluate only if targets)
        # TODO: Orientation Loss (Smooth L1 Loss applied to 6D encoding of orientation matrix; evaluate only if targets)
        #   - Make targets azimuth invariant (rotate by azimuthal position mathematically negative)
        # TODO: Velocity Loss (Smooth L1 Loss; evaluate only if targets)
        #   - Requires EGO Velocity
        #   - Make target velocity azimuth-invariant

        pass

    def training_step(self, batch, batch_idx):
        output = self.model(lidar=batch.get("lidar"), camera=batch.get("camera"), radar=batch.get("radar"))

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
