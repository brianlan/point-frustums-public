import torch


def match_target_projection_to_receptive_field(
    targets_projections: torch.Tensor,
    rf_centers: torch.Tensor,
    rf_sizes: torch.Tensor,
    layer_sizes_flat: list[int],
    base_featuremap_width: int,
    alpha: float = 1.4,
    beta: float = 0.4,
) -> torch.Tensor:
    """
    Create a binary mapping between a set of N feature vectors with corresponding receptive fields (RFs) and M projected
    targets. Returns a boolean mask of shape [N, M].

    The nonzero 2D IoU could be alternative, but it would be tricky to resolve ambiguities, and it is nontrivial how to
    set the lower IoU threshold.

    :param targets_projections: The target boxes coordinates in [x1, y1, x2, y2] format (x1 < x2, y1 < y2) [M, 4]
    :param rf_centers: The centers of the receptive fields [N, 2]
    :param rf_sizes: The sizes of the receptive fields [N, 2]
    :param layer_sizes_flat: The number of feature vectors on each featuremap layer
    :param base_featuremap_width: The width of the lowest level featuremap in the azimuthal direction
    :param alpha: Controls the upper bound of the RF size when matching (w.r.t. longer edge of the projection)
        `alpha=1`   -> Matched RFs are at most as large as targets.
        `alpha=2`   -> Matched RFs are at most twice as large as targets.
    :param beta: Controls the lower bound of the RF size when matching (w.r.t. shorter edge of the projection)
        `beta=0.5`  -> Matched RFs are at least half as large as targets.
        `beta=1`    -> Matched RFs are at least as large as targets.
    :return: Boolean mask of shape [N, M] for N feature vectors and M targets.
    """
    # Calculate the target sizes and centers
    targets_sizes = targets_projections[:, [2, 3]] - targets_projections[:, [0, 1]]
    targets_centers = (targets_projections[:, [2, 3]] + targets_projections[:, [0, 1]]) / 2

    # Evaluate distances between rf centers and target projection centers
    distances_ctr_to_ctr = torch.abs(rf_centers[:, None, :] - targets_centers[None, :, :])

    # Resolve ambiguities in the azimuthal 360° FOV
    distances_ctr_to_ctr_ambiguous = distances_ctr_to_ctr.clone()
    # Case 1: feature vector on the left side (near 0°) and target on the right side (near 360°) of the featuremap
    #   -> Shift all calculated distances by +360° and keep only the closer direction distance
    distances_ctr_to_ctr_ambiguous[:, :, 0] += base_featuremap_width
    distances_ctr_to_ctr = torch.min(distances_ctr_to_ctr, torch.abs(distances_ctr_to_ctr_ambiguous))
    # Case 2: feature vector on the right side (near 360°) and target on the left side (near 0°) of the featuremap
    #   -> Shift all calculated distances by -360° (and undo previous shift) and keep only the closer direction distance
    distances_ctr_to_ctr_ambiguous[:, :, 0] -= 2 * base_featuremap_width
    distances_ctr_to_ctr = torch.min(distances_ctr_to_ctr, torch.abs(distances_ctr_to_ctr_ambiguous))

    # Evaluate the binary mapping by selecting RFs which are at most half the size of the targets apart; note the
    #   clamping the halved target size to a minimum of 1 so that the lowest level featuremap (stride 1) is
    #   guaranteed to match
    binary_match = torch.all(torch.le(distances_ctr_to_ctr, torch.clamp(targets_sizes[None, :, :] / 2, min=1)), dim=2)

    # Condition the binary_mapping further on the sizes of the RFs
    # Scale the RFs by alpha and beta to use as lower/upper bounds
    lower_bound = rf_sizes.min(dim=1).values / alpha
    upper_bound = rf_sizes.max(dim=1).values / beta
    # Set the lower bounds smallest RF size to 0 to make sure that very small targets are not filtered out
    lower_bound[: layer_sizes_flat[0]] = 0
    # Set the upper bounds largest RF size to inf to make sure that very large targets are not filtered out
    upper_bound[-layer_sizes_flat[-1] :] = torch.inf  # NOQA: Whitespace before ":"
    # Evaluate longer edge of the target projection against the lower_bound
    binary_match &= torch.le(lower_bound[:, None], targets_sizes.max(dim=1).values[None, :])
    # Evaluate shorter edge of the target projection against the upper_bound
    binary_match &= torch.ge(upper_bound[:, None], targets_sizes.min(dim=1).values[None, :])

    return binary_match


@torch.jit.script
def sinkhorn(
    col_marginals: torch.Tensor,
    row_marginals: torch.Tensor,
    cost: torch.Tensor,
    eps: float = 1e-3,
    iter_min: int = 20,
    iter_max: int = 100,
    evaluation_frequency: int = 25,
    threshold: float = 1e-2,
) -> torch.Tensor:
    """
    Calculate transport plan in log space to avoid numerical instabilities.
    See the paragraph on numerical stability: https://cs231n.github.io/linear-classify/#softmax-classifier
    Most likely, the original implementation is this one:
    https://github.com/gpeyre/SinkhornAutoDiff/blob/1e6c8c38974a325bb59cbc5b8cb6af5787b4f58d/sinkhorn_pointcloud.py
    which is provided by the authors of many papers in the field of OT problems.
    :param col_marginals: initial targets information [M + 1] (supply)
    :param row_marginals: initial information [N] (demand)
    :param cost: initial cost matrix [N, M+1]
    :param eps:
    :param iter_min:
    :param iter_max:
    :param evaluation_frequency:
    :param threshold:
    :return:
    """
    mu = col_marginals
    nu = row_marginals
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    # Move mu and nu to log domain once
    nu.add_(1e-8).log_()
    mu.add_(1e-8).log_()

    for i in range(iter_max):
        _u = u
        v.add_(eps * (nu - torch.logsumexp(((v[:, None] + u[None, :] - cost) / eps), dim=1)))
        u = u.add(eps * (mu - torch.logsumexp(((v[:, None] + u[None, :] - cost) / eps), dim=0)))
        if i >= iter_min and i % evaluation_frequency == 0:
            if torch.norm(_u - u) < threshold:  # NOQA
                break

    # Transport plan pi = diag(a)*K*diag(b)
    pi = torch.exp((v[:, None] + u[None, :] - cost) / eps)
    return pi
