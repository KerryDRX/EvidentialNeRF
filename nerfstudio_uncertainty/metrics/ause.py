import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float


def calculate_ause(
    error1: Float[Tensor, "N 1 H W"],
    uncertainty1: Float[Tensor, "N 1 H W"],
    sqrt=True,
    normalize=False,
):
    error_vector = error1.flatten()
    uncertainty_vector = uncertainty1.flatten()
    assert len(error_vector) == len(uncertainty_vector)
    num_pixels = len(error_vector)
    removed_ratios = np.linspace(0, 1, 100, endpoint=False)
    error_sorted = torch.sort(error_vector).values
    error_sorted_by_uncertainty = error_vector[torch.sort(uncertainty_vector).indices]
    oracle_curve, uncertainty_curve = [], []
    for removed_ratio in removed_ratios:
        num_pixels_remained = int((1 - removed_ratio) * num_pixels)
        error_sorted_slice = error_sorted[:num_pixels_remained].mean()
        error_sorted_by_uncertainty_slice = error_sorted_by_uncertainty[:num_pixels_remained].mean()
        if sqrt:
            error_sorted_slice = error_sorted_slice.sqrt()
            error_sorted_by_uncertainty_slice = error_sorted_by_uncertainty_slice.sqrt()
        oracle_curve.append(error_sorted_slice.item())
        uncertainty_curve.append(error_sorted_by_uncertainty_slice.item())
    oracle_curve = np.array(oracle_curve)
    uncertainty_curve = np.array(uncertainty_curve)
    if normalize:
        oracle_curve /= oracle_curve[0]
        uncertainty_curve /= uncertainty_curve[0]
    ause = np.trapz(uncertainty_curve - oracle_curve, removed_ratios)
    return ause, oracle_curve, uncertainty_curve
