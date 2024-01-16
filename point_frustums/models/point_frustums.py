from typing import Optional
from math import prod, exp, isclose

import torch
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from point_frustums.config_dataclasses.dataset import DatasetConfig
from point_frustums.config_dataclasses.point_frustums import ModelOutputSpecification, TargetAssignment
from point_frustums.utils.geometry import (
    get_spherical_projection_boundaries,
    rotate_2d,
    rotation_matrix_from_spherical,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    iou_vol_3d,
)
from point_frustums.utils.targets import Targets
from .backbones import PointFrustumsBackbone
from .base_models import Detection3DModel
from .base_runtime import Detection3DRuntime
from .heads import PointFrustumsHead
from .necks import PointFrustumsNeck
from .training.target_assignment import match_target_projection_to_receptive_field, sinkhorn


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
        target_assignment: TargetAssignment,
        # losses: LossesConfig,
        **kwargs,
    ):
        super().__init__(*args, model=model, dataset=dataset, **kwargs)
        self.target_assignment = target_assignment
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

        discretize = self.model.backbone.lidar.discretize
        strides = self.model.neck.fpn.strides.values()

        size_per_layer = []
        size_per_layer_flat = []
        feat_rfs_center = []
        for layer, stride in self.model.neck.fpn.strides.items():
            assert isclose(discretize.n_splits_pol % stride[0], 0), (
                f"The polar input featuremap resolution {discretize.n_splits_pol} is not divisible by the stride "
                f"{stride[0]} on layer {layer}."
            )
            assert isclose(discretize.n_splits_azi % stride[1], 0), (
                f"The azimuthal input featuremap resolution {discretize.n_splits_azi} is not divisible by the "
                f"stride {stride[1]} on layer {layer}."
            )
            size_per_layer.append((int(discretize.n_splits_pol / stride[0]), int(discretize.n_splits_azi / stride[1])))
            size_per_layer_flat.append(int(discretize.n_splits / prod(stride)))

            # Get the node position w.r.t. the original input data shape
            nodes_pol = torch.arange(0, discretize.n_splits_pol - 1e-8, step=stride[0])
            nodes_azi = torch.arange(0, discretize.n_splits_azi - 1e-8, step=stride[1])

            # Create meshgrid representation for easy indexing
            nodes_pol, nodes_azi = torch.meshgrid(nodes_pol, nodes_azi, indexing="xy")
            nodes_pol, nodes_azi = nodes_pol.flatten(), nodes_azi.flatten()
            # Merge to obtain the [x, y] coordinates of the centers
            feat_rfs_center.append(torch.stack((nodes_azi, nodes_pol), dim=1))
            # size_per_layer.append(feat_center_grid[-1].shape[0])
        feat_rfs_center = torch.cat(feat_rfs_center, dim=0)
        self.register_buffer("feat_rfs_centers", feat_rfs_center)

        feat_rfs = reference_boxes_centered.repeat_interleave(torch.tensor(size_per_layer_flat), dim=0)
        feat_rfs += feat_rfs_center.repeat(1, 2)
        self.register_buffer("feat_rfs", feat_rfs)

        feat_rfs_sizes = torch.tensor(layers_rfs).repeat_interleave(torch.tensor(size_per_layer_flat), dim=0)
        self.register_buffer("feat_rfs_sizes", feat_rfs_sizes)

        # Scale-scale-shift grid centers to angular center coordinates
        feat_center_pol_azi = feat_rfs_center[:, [1, 0]]
        feat_center_pol_azi.div_(torch.tensor([discretize.n_splits_pol, discretize.n_splits_azi]))
        feat_center_pol_azi.mul_(torch.tensor([discretize.range_pol, discretize.range_azi]))
        feat_center_pol_azi.add_(torch.tensor([discretize.fov_pol[0], discretize.fov_azi[0]]))
        self.register_buffer("feat_center_pol_azi", feat_center_pol_azi)

        return ModelOutputSpecification(
            strides=list(strides),
            layer_sizes=size_per_layer,
            layer_sizes_flat=size_per_layer_flat,
            total_size=sum(size_per_layer_flat),
        )

    @property
    def receptive_fields(self):
        return self.get_buffer("feat_rfs")

    @property
    def receptive_fields_sizes(self):
        return self.get_buffer("feat_rfs_sizes")

    @property
    def receptive_fields_centers(self):
        return self.get_buffer("feat_rfs_centers")

    @property
    def feature_vectors_angular_centers(self):
        return self.get_buffer("feat_center_pol_azi")
    
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

    @staticmethod
    def _one_hot(label: torch.Tensor, num_classes: int, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert label.dim() == 1
        one_hot = F.one_hot(label.clip(min=0), num_classes=num_classes)  # pylint: disable=not-callable
        one_hot = torch.where(torch.ge(label[:, None], 0), one_hot, 0)
        if index is not None:
            one_hot = one_hot[index, ...]
        return one_hot.float()

    def _encode_center(self, x: torch.Tensor, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the M spherical center coordinates w.r.t. the angular position of the feature vectors.
        :param x:
        :param idx_feat:
        :return:
        """
        center_pol_azi = self.feat_center_pol_azi
        if idx_feat is not None:
            center_pol_azi = center_pol_azi[idx_feat, :]
        x = x.clone()
        x[..., [1, 2]] -= center_pol_azi
        x[..., 1] /= self.model.backbone.lidar.discretize.delta_pol
        x[..., 2] /= self.model.backbone.lidar.discretize.delta_azi
        return x

    def _decode_center(self, x: torch.Tensor, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode the M spherical center coordinates w.r.t. the angular position of the feature vectors.
        :param x:
        :param idx_feat:
        :return:
        """
        center_pol_azi = self.feat_center_pol_azi
        if idx_feat is not None:
            x = x[..., idx_feat, :]
            center_pol_azi = center_pol_azi[idx_feat, :]
        x = x.clone()
        x[..., 1] *= self.model.backbone.lidar.discretize.delta_pol
        x[..., 2] *= self.model.backbone.lidar.discretize.delta_azi
        x[..., [1, 2]] += center_pol_azi
        return x

    @staticmethod
    def _encode_wlh(x: torch.Tensor) -> torch.Tensor:
        return x.log()

    @staticmethod
    def _decode_wlh(x: torch.Tensor, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if idx_feat is not None:
            x = x[..., idx_feat, :]
        return x.exp()

    def _encode_orientation(self, x: torch.Tensor, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode the provided M orientation matrices w.r.t. the angular coordinates on the featuremap and subsequently in
        form of the 6D encoding suggested in: http://arxiv.org/abs/1812.07035
        :param x: The orientation matrices [M, 3, 3]
        :param idx_feat: The indices of the feature vectors corresponding to the provided orientations [M]
        :return:
        """
        theta, phi = self.feature_vectors_angular_centers.unbind(dim=-1)
        if idx_feat is not None:
            theta = theta[idx_feat]
            phi = phi[idx_feat]
        # Transpose along the last 2 dimensions to invert the direction of rotation
        rotation_matrices = rotation_matrix_from_spherical(theta, phi).transpose(-1, -2)
        x = torch.einsum("...ij,...jk->...ik", rotation_matrices, x)
        return matrix_to_rotation_6d(x)

    def _decode_orientation(self, x: torch.Tensor, idx_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode the provided M vectors encoded in form of the 6D encoding (http://arxiv.org/abs/1812.07035) to the
        matrix form and then rotate them back according to the angular coordinates on the featuremap.
        :param x: The encoded representation of the orientation [M, 6]
        :param idx_feat: The indices of the feature vectors corresponding to the provided orientations [M]
        :return:
        """
        theta, phi = self.feature_vectors_angular_centers.unbind(dim=-1)
        if idx_feat is not None:
            x = x[..., idx_feat, :]
            theta = theta[idx_feat]
            phi = phi[idx_feat]
        x = rotation_6d_to_matrix(x)
        rotation_matrices = rotation_matrix_from_spherical(theta, phi)
        return torch.einsum("...ij,...jk->...ik", rotation_matrices, x)

    def _encode_velocity(
        self, x: torch.tensor, ego_velocity: torch.Tensor, idx_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode the velocity as [normal, tangential] velocity by rotating it by the negative of the azimuthal position.
        :param x:
        :param ego_velocity:
        :param idx_feat:
        :return:
        """
        x = x - ego_velocity
        phi = self.feature_vectors_angular_centers[:, 1]
        if idx_feat is not None:
            phi = phi[idx_feat]
        # Invert the rotation by negating the sign of phi, this is possible only in the 2D case
        return rotate_2d(phi=-phi, x=x[..., [0, 1]])

    def _decode_velocity(
        self, x: torch.Tensor, ego_velocity: torch.Tensor, idx_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode the [normal, tangential] velocity into cartesian [x, y] components by rotating it by the azimuthal
        position.
        :param x:
        :param ego_velocity:
        :param idx_feat:
        :return:
        """
        phi = self.feature_vectors_angular_centers[:, 1]
        if idx_feat is not None:
            x = x[..., idx_feat, :]
            phi = phi[idx_feat]

        x = rotate_2d(phi=phi, x=x)
        return x + ego_velocity

    def assign_target_index_to_features(
        self,
        targets: Targets,
        target_labels: torch.Tensor,
        target_corners: torch.Tensor,
        feat_labels: torch.Tensor,
        feat_centers: torch.Tensor,
        feat_corners: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assign a target to each feature vector of the concatenated featuremaps. Apply to a single sample from the batch.
        :param targets:
        :param target_corners:
        :param target_labels:
        :param feat_labels:
        :param feat_centers:
        :param feat_corners:
        :return: An integer tensor where -1 represents background and all values >= 0 correspond to one of the targets.
        """
        if target_labels is None:
            return torch.full((self.featuremap_parametrization.total_size,), -1, device=self.device)

        targets_spherical_projection = get_spherical_projection_boundaries(
            centers=targets.center,
            wlh=targets.wlh,
            orientation=targets.orientation,
            fov_pol=self.discretization.fov_pol,
            fov_azi=self.discretization.fov_azi,
            delta_pol=self.discretization.delta_pol,
            delta_azi=self.discretization.delta_azi,
        )
        binary_pre_mapping = match_target_projection_to_receptive_field(
            targets_projections=targets_spherical_projection,
            rf_centers=self.receptive_fields_centers,
            rf_sizes=self.receptive_fields_sizes,
            layer_sizes_flat=self.featuremap_parametrization.layer_sizes_flat,
            base_featuremap_width=self.model.backbone.lidar.discretize.n_splits_azi,
        )

        # Get the indices that map between feature vectors and targets
        #   Column 0: Indices of feature vectors, use to obtain a subset of predictions that can be mapped to targets
        #   Column 1: Indices of targets, use to broadcast targets to the assigned subset of feature vectors
        idx = torch.nonzero(binary_pre_mapping).unbind(-1)

        # Evaluate pairwise IoU between the matched subset of predictions corresponding targets
        iou = torch.zeros_like(binary_pre_mapping, dtype=torch.float).index_put_(
            idx, iou_vol_3d(feat_corners[idx[0], ...], target_corners[idx[1], ...])[0]
        )
        # Initialize the center distance to all zeros
        center_distance = torch.zeros_like(binary_pre_mapping).float()
        # Calculate the cartesian distances between predictions and mapped targets (prev.: the distance of radius)
        center_distance[idx] = (feat_centers[idx[0], :] - targets.center[idx[1], :]).norm(dim=-1)
        # Normalize the distance by a factor based on the target box size
        center_distance[idx] = center_distance[idx].div(targets.wlh[idx[1], :].norm(dim=-1))
        # Squash the relative distance to the range [0, 1) by application of tanh
        #   The division by the empirical factor kappa shifts the relevant range into dynamic range of tanh
        center_distance[idx] = center_distance[idx].div(self.target_assignment.kappa).tanh()

        # Broadcast one-hot encoded labels s.t. the first 2 dimensions match the shape of the binary mapping and
        # evaluate the focal loss
        cost_classification: torch.Tensor = sigmoid_focal_loss(
            feat_labels[:, None, :].expand(binary_pre_mapping.shape + (self.dataset.annotations.n_classes,)),
            target_labels[None, :, :].expand(binary_pre_mapping.shape + (self.dataset.annotations.n_classes,)),
            reduction="none",
        ).sum(dim=-1)

        # Evaluate the focal loss against the all-zero labels to use as classification cost for the supplementary
        # background class
        background_labels = torch.zeros_like(feat_labels)
        cost_background = sigmoid_focal_loss(feat_labels, background_labels, reduction="none").sum(dim=-1).tanh()
        # Shift by the factors beta and gamma to ensure that the pre-assigned background will have no smaller cost than
        # any point on the foreground ever
        cost_background += self.target_assignment.beta + self.target_assignment.gamma

        # Combine classification/regression and background cost
        cost_classification = cost_classification.tanh()
        min_fg_cost = cost_classification[binary_pre_mapping].max()
        cost_classification[~binary_pre_mapping] = torch.clamp(
            cost_classification[~binary_pre_mapping], min=min_fg_cost + 1e-2
        )
        # Initialize to the bias which ensures that the background is never assigned lower cost than the foreground
        cost = (self.target_assignment.beta + self.target_assignment.gamma) * (~binary_pre_mapping).float()
        # Add the scaled classification cost
        cost += self.target_assignment.alpha * cost_classification.tanh()
        # Add the scaled IoU cost
        cost += self.target_assignment.beta * (1 - iou)
        # Add the scaled center distance cost
        cost += self.target_assignment.gamma * center_distance
        # Concatenate the pre-determined background class cost
        cost = torch.cat((cost, cost_background[:, None]), dim=1)

        # Set the demand to ones as each prediction requires to be assigned to exactly one class (object or background)
        demand = torch.ones((self.featuremap_parametrization.total_size,), device=self.device).float()
        # Based on the IoU, evaluate how many predictions each of the targets can supply, constrain by min_k and top_k
        supply = torch.topk(iou, dim=0, k=self.target_assignment.min_k).values.sum(dim=0)
        supply = torch.clamp(supply, min=self.target_assignment.min_k)
        # Append the unassigned supply for the background class (note: tensor.sum returns a 0D tensor)
        unassigned_supply = self.featuremap_parametrization.total_size - supply.sum()
        supply = torch.cat((supply, unassigned_supply[None]))

        # Evaluate the transport plan PI of the optimal transport problem (Comprehensive read: arXiv:1803.00567 p 62-84)
        # The implemented algorithm is a version by renown researchers from the field but tricky to derive
        pi = sinkhorn(
            cost=cost,
            row_marginals=demand,
            col_marginals=supply,
            eps=self.target_assignment.epsilon,
            threshold=self.target_assignment.threshold,
            iter_max=self.target_assignment.max_iter,
        )

        # Rescale s.t. the maximum pi for each target equals 1.
        scale_factor, _ = pi.max(dim=0)
        pi = pi / scale_factor[None, :]
        # Select the target with the highest value of PI for each patch (if the last row -> background)
        _, match_index = pi.max(dim=1)

        # Assign entries that matched the last/background row to the background i.e. -1
        foreground_index = match_index != binary_pre_mapping.shape[1]
        match_index[~foreground_index] = -1

        return match_index

    def training_step(self, batch, batch_idx):
        output = self.model(lidar=batch.get("lidar"), camera=batch.get("camera"), radar=batch.get("radar"))

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
