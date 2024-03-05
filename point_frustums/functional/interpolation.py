from typing import Literal, Optional

import torch


def interpolate_to_support(
    x: torch.Tensor,
    y: torch.Tensor,
    support: torch.Tensor,
    left: Optional[float] = None,
    right: Optional[float] = None,
    order: Literal["increasing", "decreasing"] = "increasing",
) -> torch.Tensor:
    """
    Rudimentary implementation of numpy.interp for the 1D case.
    :param x: The original coordinates, must be sorted and unique.
    :param y: The original values.
    :param support: The support points to which y shall be interpolated, must be sorted and unique.
    :param left: The padding used on the left side of min(x), if None, repeats the leftmost interpolated y
    :param right: The padding used on the right side of max(x), if None, repeats the rightmost interpolated y
    :param order: If decreasing, all input is reversed to interpolate correctly and then flipped again before returning.
    :return:
    """
    assert order in ("increasing", "decreasing")

    if x.numel() == 0:
        return torch.zeros_like(support)

    if order == "decreasing":
        x = x.flip(dims=(0,))
        y = y.flip(dims=(0,))
        support = support.flip(dims=(0,))
        left, right = right, left

    assert torch.ge(x.diff(), 0).all().item(), "The input x must be (reverse) sorted."

    # Evaluate the forward difference (except the right edge point as it cannot be evaluated)
    slope = torch.zeros_like(x)
    num, den = (y[1:] - y[:-1]), (x[1:] - x[:-1])
    mask = ~torch.isclose(den, torch.zeros_like(num))
    slope[:-1] = torch.where(mask, num / den, num)

    # Evaluate which of the support points are within the range of x
    x_lower_bound = torch.ge(support, x.min())
    x_upper_bound = torch.le(support, x.max())
    support_nonzero_mask = x_lower_bound & x_upper_bound
    # Subset the support points accordingly
    support_nonzero = support[support_nonzero_mask]
    # If no support points fall inside the range of x, return all-zero

    # Get the indices of the closest point to the left for each support point
    support_insert_indices = torch.searchsorted(x, support_nonzero, side="right") - 1
    # Get the offset from the point to the left to the support point
    support_nonzero_offset = support_nonzero - x[support_insert_indices]
    # Calculate the value for the nonzero support: value of the point to the left plus slope times offset
    support_nonzero_values = y[support_insert_indices] + slope[support_insert_indices] * support_nonzero_offset

    # Create the output tensor and place the nonzero support
    support_values = torch.empty_like(support)
    support_values[support_nonzero_mask] = support_nonzero_values

    # Pad values to the left and right of the x-range
    left_padding = left if left is not None else y[0]
    right_padding = right if right is not None else y[-1]
    left_index, right_index = x_lower_bound.nonzero()[0], x_upper_bound.nonzero()[-1]
    support_values[:left_index] = left_padding
    support_values[right_index + 1 :] = right_padding  # NOQA: Whitespace before ':'

    if order == "decreasing":
        support_values = support_values.flip(dims=(0,))

    return support_values
