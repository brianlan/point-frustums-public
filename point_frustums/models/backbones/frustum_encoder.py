from collections.abc import Mapping, Sequence
from dataclasses import asdict
from functools import cached_property
from math import isclose
from typing import Optional, Literal, Dict

import torch
from torch import nn
from xformers.ops.fmha import attn_bias, memory_efficient_attention

from point_frustums.config_dataclasses.point_frustums import (
    ConfigDiscretize,
    ConfigVectorize,
    ConfigDecorate,
    ConfigReduce,
    ConfigTransformerFrustumEncoder,
)


@torch.jit.script
def symmetrize_max(  # pylint: disable=too-many-arguments
    pc: torch.Tensor,
    n_frustums: int,
    i_frustum: torch.Tensor,
    i_inv: torch.Tensor,
    counts_padded: torch.Tensor,
    context: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the max via the torch.scatter_reduce OP.
    :param pc:
    :param n_frustums:
    :param i_frustum:
    :param i_inv:
    :param counts_padded:
    :param context:
    :return:
    """
    n_points, n_channels = pc.shape
    # Prepare the output tensor and the index tensor for the scatter op
    i_frustum = i_frustum[:, None].expand(n_points, n_channels)
    # Initialize the output tensor with zeros and fill by the maximum value per frustum
    frustum_max = pc.new_zeros((n_frustums, n_channels))
    frustum_max = torch.scatter_reduce(input=frustum_max, dim=0, index=i_frustum, reduce="amax", src=pc)
    return frustum_max, context


@torch.jit.script
def symmetrize_mean(  # pylint: disable=too-many-arguments
    pc: torch.Tensor,
    n_frustums: int,
    i_frustum: torch.Tensor,
    i_inv: torch.Tensor,
    counts_padded: torch.Tensor,
    context: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the mean via the torch.scatter_add_ OP as method to compute the frustum-wise sum.
    :param pc:
    :param n_frustums:
    :param i_frustum:
    :param i_inv:
    :param counts_padded:
    :param context:
    :return:
    """

    if "mean" in context:
        return context["mean"], context

    n_points, n_channels = pc.shape
    # Prepare the output tensor and the index tensor for the scatter op
    i_frustum = i_frustum[:, None].expand(n_points, n_channels)
    # Initialize the output tensor with zeros and fill by the maximum value per frustum
    mean = pc.new_zeros((n_frustums, n_channels))
    mean = torch.scatter_add(input=mean, dim=0, index=i_frustum, src=pc)

    mask = counts_padded != 0
    mean[mask, :] = mean[mask, :].div(counts_padded[mask, None])

    context["mean"] = mean
    return mean, context


@torch.jit.script
def symmetrize_std(  # pylint: disable=too-many-arguments
    pc: torch.Tensor,
    n_frustums: int,
    i_frustum: torch.Tensor,
    i_inv: torch.Tensor,
    counts_padded: torch.Tensor,
    context: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the std via the torch.scatter_add OP as method to compute the frustum-wise sum.
    :param pc:
    :param n_frustums:
    :param i_frustum:
    :param i_inv:
    :param counts_padded:
    :param context:
    :return:
    """
    if "std" in context:
        return context["std"], context

    if "mean" not in context:
        _, context = symmetrize_mean(
            pc=pc, n_frustums=n_frustums, i_frustum=i_frustum, i_inv=i_inv, counts_padded=counts_padded, context=context
        )
    mean = context["mean"]
    n_points, n_channels = pc.shape

    pc = pc.sub(mean[i_frustum, :][i_inv, :])
    pc = pc.square()

    std = pc.new_zeros((n_frustums, n_channels))
    std = torch.scatter_add(input=std, dim=0, index=i_frustum[:, None].expand(n_points, n_channels), src=pc)

    mask = torch.ne(counts_padded, 0)
    std[mask, :] = std[mask, :].div(counts_padded[mask, None])
    mask = torch.gt(std, 0.0)
    std[mask] = std[mask].sqrt()

    context["std"] = std
    return std, context


class TransformerFrustumEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    class FeedForward(nn.Module):
        def __init__(self, n_channels, expansion_factor: int = 4, dropout: float = 0.1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(n_channels, expansion_factor * n_channels),
                nn.GELU(),
                nn.Linear(expansion_factor * n_channels, n_channels),
            )
            self.norm = nn.LayerNorm(n_channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x) -> torch.Tensor:
            return self.dropout(self.network(self.norm(x)) + x)

    def __init__(  # pylint: disable=too-many-arguments
        self,
        n_splits_azi: int,
        n_splits_pol: int,
        n_channels_in: int,
        n_channels_embedding: int,
        n_channels_projection: int,
        n_heads: int = 1,
        n_encoders: int = 1,
        forward_expansion: int = 4,
        dropout: float = 0.1,
    ):
        """
        Enrich points with information from the local context by applying a sequence of n encoder blocks. The
        semantically richer pointcloud is then projected to the output space and reduced to a pseudo-image by applying
        cross attention with a learnable query/decoder parameter.

        The TransformerFrustumEncoder makes extensive use of the xFormers library to reduce the memory and compute
        footprint of the self attention mechanism.
        :param n_splits_azi:
        :param n_splits_pol:
        :param n_channels_in:
        :param n_channels_embedding:
        :param n_channels_projection:
        :param n_heads:
        :param n_encoders:
        :param forward_expansion:
        :param dropout:
        """
        super().__init__()

        self.n_splits_azi = n_splits_azi
        self.n_splits_pol = n_splits_pol
        self.n_splits = n_splits_azi * n_splits_pol

        self.n_channels_in = n_channels_in
        self.n_channels_embedding = n_channels_embedding
        self.n_channels_projection = n_channels_projection

        self.n_heads = n_heads
        self.n_encoders = n_encoders
        self.forward_expansion = forward_expansion
        self.dropout = dropout

        self.n_channels_per_head = self.n_channels_embedding // self.n_heads
        assert isclose(self.n_channels_embedding % self.n_heads, 0), "Embedding dimension not divisible by `n_heads`."

        self.fc_pre = None
        if self.n_channels_in != self.n_channels_embedding:
            self.fc_pre = nn.Linear(self.n_channels_in, self.n_channels_embedding)

        # TODO: The below could be merged into a Sequential of modules if it were not for the mask (Sequential allows
        #  only one input arg) -> Put it in a tuple?
        self.qvk_self_attention = nn.ModuleList()
        self.ffn_self_attention = nn.ModuleList()
        self.norm_self_attention = nn.ModuleList()
        for _ in range(self.n_encoders):
            self.qvk_self_attention.append(nn.Linear(self.n_channels_per_head, 3 * self.n_channels_per_head))
            self.ffn_self_attention.append(self.FeedForward(n_channels=self.n_channels_per_head, dropout=dropout))
            self.norm_self_attention.append(nn.LayerNorm(self.n_channels_per_head))

        self.kv_cross_attention = nn.Linear(self.n_channels_embedding, 2 * self.n_channels_projection)
        self.decoder = nn.Parameter(torch.empty(1, self.n_splits_pol, 1, self.n_channels_projection))
        self.ffn_cross_attention = self.FeedForward(n_channels=self.n_channels_projection, dropout=dropout)
        self.norm_cross_attention = nn.LayerNorm(self.n_channels_projection)

    def forward(  # pylint: disable=too-many-locals
        self,
        pc: torch.Tensor,
        i_unique: torch.Tensor,
        counts: list[int],
    ) -> torch.Tensor:
        """
        Enrich the pointcloud with the local context and then apply a reduction to the pseudo-image by performing cross
        attention with the decoder query parameter.

        The cross attention mechanism is equivalent to feeding the pointcloud through a linear layer without bias and
        subsequently applying average pooling within frustums. One notable detail is, that the implemented algorithm is
        azimuth-invariant but uses a distinct projection space for the polar dimension (rows).
        :param pc:
        :param i_unique:
        :param counts:
        :return:
        """
        # For brevity, the comments use:
        #   N := n_points
        #   M := n_frustums
        #   I := len(i_unique) (less than M, the number of non-empty frustums)
        #   P := n_splits_pol
        #   H := n_heads
        #   F := n_channels_in
        #   E := n_channels_embedding
        #   K := n_channels_per_head
        #   L := n_channels_projection
        n_points, _ = pc.shape
        # Apply preceding layer to pointcloud to get the correct embedding size s.t. residual connections are possible.
        if self.fc_pre is not None:
            pc = self.fc_pre(pc)

        # SELF ATTENTION (THE POINTCLOUD IS ENRICHED WITH THE FRUSTUM CONTEXT)
        # Create BlockDiagonalMask to perform sparse self attention
        mask = attn_bias.BlockDiagonalMask.from_seqlens(counts, counts)

        # Preprocess PC dimension for encoder blocks to be explicit w.r.t. the number of heads [1, N, H, K]
        pc = pc.reshape([1, n_points, self.n_heads, self.n_channels_per_head])

        # Iteratively feed PC through layers
        for qkv, ffn, norm in zip(self.qvk_self_attention, self.ffn_self_attention, self.norm_self_attention):
            # Normalize and then encode PC as query, key and value [1, N, H, K]
            q, k, v = qkv(norm(pc)).reshape([1, n_points, self.n_heads, 3, self.n_channels_per_head]).unbind(-2)
            # Evaluate self-attention and residual connection [1, N, H, K]
            pc = memory_efficient_attention(query=q, key=k, value=v, attn_bias=mask, p=self.dropout) + pc
            # Apply FFN (norm, fully connected network, residual connection, dropout)
            pc = ffn(pc)
        # Stack heads
        pc = pc.reshape([1, n_points, self.n_channels_embedding])

        # CROSS ATTENTION (DECODER QUERIES THE ENRICHED POINTCLOUD)
        # Create BlockDiagonalMask that links nonempty frustums to all the points assigned to the latter.
        mask = attn_bias.BlockDiagonalMask.from_seqlens(len(i_unique) * [1], counts)
        # Normalize and then encode PC as key and value [1, N, 1, L]
        k, v = (
            self.kv_cross_attention(self.norm_cross_attention(pc))
            .reshape([1, n_points, 1, 2, self.n_channels_projection])
            .unbind(-2)
        )
        # Initialize output tensor with zeros
        feats = pc.new_zeros((self.n_frustums, self.n_channels_projection))
        # Repeat the first dimension of the decoder so that it can be indexed with the global frustum index (row-major)
        # [1, P, 1, L] -> [1, M, 1, L]
        # https://discuss.pytorch.org/t/repeat-a-nn-parameter-for-efficient-computation/25659/6
        decoder = self.decoder.repeat(1, self.n_splits_azi, 1, 1)
        # Index-select to keep only the parameters corresponding to non-empty frustums [1, M, 1, L] -> [1, I, 1, L]
        decoder = decoder[:, i_unique, :, :]  # pylint: disable=unsubscriptable-object
        # Evaluate cross-attention and residual connection
        # Query: [1, I, 1, L] | Key, Value: [1, N, 1, L] -> Attention Matrix: [I, N] -> Output: [1, I, 1, L]
        out = memory_efficient_attention(query=decoder, key=k, value=v, attn_bias=mask, p=self.dropout)
        # Index-put the output of the cross attention into the output tensor [1, I, 1, L] -> [1, M, 1, L]
        feats[i_unique, :] = (out + decoder)[0, :, 0, :]  # TODO: Discuss whether this residual connection is sensible

        # Apply feed forward network (norm, fully connected network, residual connection, dropout)
        return self.ffn_cross_attention(feats)


def _create_ffn(n_channels: int, layers: Sequence[int], dropout: float = 0.1):
    network = []
    for layer in layers:
        network.append(nn.Linear(n_channels, layer))
        network.append(nn.GELU())
        n_channels = layer
    network.append(nn.LayerNorm(n_channels))
    network.append(nn.Dropout(dropout))
    return nn.Sequential(*network)


def decorator_distance_to_mean(
    pc_channel: torch.Tensor,
    n_frustums: int,
    i_frustum: torch.Tensor,
    i_inv: torch.Tensor,
    std: Optional[float] = 1.0,
):
    """
    :param pc_channel:
    :param n_frustums:
    :param i_frustum:
    :param i_inv:
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :return:
    """
    mean = torch.scatter_add(pc_channel.new_zeros((n_frustums,)), dim=0, index=i_frustum, src=pc_channel)
    # Subset to the number of non-empty frustums (i_frustum) and then map the frustum mean value to the points (i_inv)
    pc_channel = pc_channel - mean[i_frustum][i_inv]
    return (pc_channel / std)[:, None]


def decorator_relative_angle(pc_channel: torch.Tensor, delta_rad: float, std: Optional[float] = 1.0):
    """
    Returns the (rad) angle between the center of the frustum and the point in the respective dimension.
    :param pc_channel:
    :param delta_rad:
    :param std: The normalization constant that scales the standard deviation to 1, determined empirically
    :return:
    """
    # Normalize output by shifting to the frustum center and dividing by an (empirically estimated) standard deviation
    angle = (torch.remainder(pc_channel, delta_rad) - (delta_rad / 2)) / delta_rad
    angle = angle / std
    return angle[:, None]


class FrustumEncoder(nn.Module):  # pylint: disable=too-many-instance-attributes
    symmetric_functions = {"max": symmetrize_max, "mean": symmetrize_mean, "std": symmetrize_std}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        channels_in: list[str, ...],
        discretize: ConfigDiscretize,
        decorate: ConfigDecorate,
        vectorize: ConfigVectorize,
        symmetrize: Sequence[Literal["max", "mean", "std"]],
        reduce: ConfigReduce,
        transformer_frustum_encoder: Optional[ConfigTransformerFrustumEncoder] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels_in = channels_in
        self.discretize = discretize
        self.decorate = decorate
        self.vectorize = vectorize
        self.symmetrize = symmetrize
        self.reduce = reduce
        self.tfe = transformer_frustum_encoder
        self.dropout = dropout

        if self.tfe is not None:
            self.tfe = TransformerFrustumEncoder(**asdict(self.tfe))

        self.decorator_keep_channels = self.get_decorator_keep_channels()

        self.ffn_vectorize = _create_ffn(self.n_channels["decorated"], self.vectorize.layers, dropout=dropout)
        self.ffn_reduce = _create_ffn(self.n_channels["squashed"], self.reduce.layers, dropout=dropout)

    @cached_property
    def n_channels(self) -> Mapping:
        stages = {"input": len(self.channels_in)}

        # Determine the output dimensionality of the decorator
        if self.decorate.n_channels_out is not None:
            stages["decorated"] = self.decorate.n_channels_out
        else:
            stages["decorated"] = stages["input"]

        # Determine the output dimensionality of the vectorizer
        if self.tfe is not None and self.vectorize.n_channels_out is not None:
            assert self.tfe.n_channels_projection == self.vectorize.n_channels_out
            stages["squashed"] = self.vectorize.n_channels_out
        elif self.vectorize.n_channels_out is not None:
            stages["squashed"] = self.vectorize.n_channels_out
        elif self.tfe is not None:
            stages["squashed"] = self.tfe.n_channels_projection
        else:
            stages["squashed"] = stages["decorated"]

        # Determine how many symmetric functions are applied to the vectorized pointcloud
        n_symmetric_functions = len(self.symmetrize)
        if self.tfe is not None:
            n_symmetric_functions += 1
        stages["squashed"] *= n_symmetric_functions

        # Determine the output dimensionality of the dimension reduction
        if self.reduce.n_channels_out is not None:
            stages["reduced"] = self.reduce.n_channels_out
        else:
            stages["reduced"] = stages["squashed"]
        return stages

    def get_decorator_keep_channels(self):
        """Evaluates the indices of the input channels that are kept after applying the decorators in correct order."""
        channels = []
        keep_channels = set(self.channels_in).intersection(self.decorate.channels_out)
        for channel in self.channels_in:
            if channel in keep_channels:
                channels.append(self.channels_in.index(channel))
        return channels

    def filter_fov(self, pc: torch.Tensor) -> torch.Tensor:
        channel_index_azi, channel_index_pol = self.channels_in.index("azimuthal"), self.channels_in.index("polar")
        mask = (
            (pc[:, channel_index_azi] >= self.discretize.fov_azi[0])
            & (pc[:, channel_index_azi] < self.discretize.fov_azi[1])
            & (pc[:, channel_index_pol] >= self.discretize.fov_pol[0])
            & (pc[:, channel_index_pol] < self.discretize.fov_pol[1])
        )
        return pc[mask, ...]

    def get_frustum_index(self, pc: torch.Tensor):
        channel_index_azi, channel_index_pol = self.channels_in.index("azimuthal"), self.channels_in.index("polar")
        i_azi = ((pc[:, channel_index_azi] - self.discretize.fov_azi[0]) // self.discretize.delta_azi).to(torch.int64)
        i_pol = ((pc[:, channel_index_pol] - self.discretize.fov_pol[0]) // self.discretize.delta_pol).to(torch.int64)
        return i_azi + i_pol * self.discretize.n_splits_azi

    @staticmethod
    def sort_by_frustum_index(pc: torch.Tensor, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices, sorting_indices = torch.sort(indices)
        pc = pc[sorting_indices, :]
        return pc, indices

    def get_frustum_stats(self, indices):
        i_unique, i_inv, counts = indices.unique_consecutive(return_inverse=True, return_counts=True)
        counts_padded = counts.new_zeros((self.discretize.n_splits,))
        counts_padded[i_unique] = counts
        return i_unique, i_inv, counts, counts_padded

    def apply_decorators(self, pc: torch.Tensor, i_frustum: torch.Tensor, i_inv: torch.Tensor) -> torch.Tensor:
        decorated_channels = []
        for fn in self.decorate.functions:
            pc_channel = pc[:, self.channels_in.index(fn.channel)]
            match fn.id:
                case "relative_angle":
                    delta_rad = {"azimuthal": self.discretize.delta_azi, "polar": self.discretize.delta_pol}[fn.channel]
                    decorated_channels.append(
                        decorator_relative_angle(
                            pc_channel=pc_channel,
                            delta_rad=delta_rad,
                            std=fn.std,
                        )
                    )
                case "distance_to_mean":
                    decorated_channels.append(
                        decorator_distance_to_mean(
                            pc_channel=pc_channel,
                            n_frustums=self.discretize.n_splits,
                            i_frustum=i_frustum,
                            i_inv=i_inv,
                            std=fn.std,
                        )
                    )
                case _:
                    raise NotImplementedError(f"Decorator {fn.id} not implemented.")
        return torch.cat((pc[:, self.decorator_keep_channels], *decorated_channels), dim=1)

    def apply_symmetric_functions(  # pylint: disable=too-many-arguments
        self,
        pc: torch.Tensor,
        n_frustums: int,
        i_frustum: torch.Tensor,
        i_inv: torch.Tensor,
        counts_padded: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        pseudo_images = []
        context = {}
        for function in self.symmetrize:
            pseudo_image, context = self.symmetric_functions[function](
                pc=pc,
                n_frustums=n_frustums,
                i_frustum=i_frustum,
                i_inv=i_inv,
                counts_padded=counts_padded,
                context=context,
            )
            pseudo_images.append(pseudo_image)
        return pseudo_images

    def vectorize_and_squash(  # pylint: disable=too-many-arguments
        self,
        pc: torch.Tensor,
        i_frustum: torch.Tensor,
        i_unique: torch.Tensor,
        i_inv: torch.Tensor,
        counts: torch.Tensor,
        counts_padded: torch.Tensor,
    ) -> torch.Tensor:
        # If the number of symmetric functions (other than the TFE is nonzero, apply vectorization and symmetric fn
        featuremap = []
        if len(self.symmetrize) > 0:
            # Apply PointNet-like encoder to entire PC and then reduce by symmetric function(s)
            featuremap.extend(
                self.apply_symmetric_functions(
                    pc=self.ffn_vectorize(pc),
                    n_frustums=self.n_splits,
                    i_frustum=i_frustum,
                    i_inv=i_inv,
                    counts_padded=counts_padded,
                )
            )
        if self.tfe is not None:
            # Apply the TFE to the pc to enrich points with local context and project to output dimension
            featuremap.append(self.tfe(pc=pc, i_unique=i_unique, counts=counts.tolist()))

        return torch.cat(featuremap, dim=1)

    def forward(self, batch: list[torch.Tensor]) -> torch.Tensor:
        batch_size = len(batch)
        features = batch[0].new_empty((batch_size, self.discretize.n_splits, self.n_channels["squashed"]))

        for i, pc in enumerate(batch):
            pc = self.filter_fov(pc)
            i_frustum = self.get_frustum_index(pc)
            pc, i_frustum = self.sort_by_frustum_index(pc, i_frustum)
            i_unique, i_inv, counts, counts_padded = self.get_frustum_stats(i_frustum)
            # Apply decorators to entire PC
            decorated = self.apply_decorators(pc, i_frustum=i_frustum, i_inv=i_inv)
            # Squash irregular number of points to regular array by applying symmetric function
            # PyTorch automatically assigns tensors C-contiguous which matches the creation of the global index
            features[i, :, :] = self.vectorize_and_squash(decorated, i_frustum, i_unique, i_inv, counts, counts_padded)

        # Apply dimension reduction to the featuremap
        features = self.ffn_reduce(features)
        # Reshape featuremap to form a pseudo image
        features = features.permute(0, 2, 1).reshape(
            batch_size,
            self.n_channels["reduced"],
            self.discretize.n_splits_pol,
            self.discretize.n_splits_azi,
        )
        return features
