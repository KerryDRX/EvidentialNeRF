import numpy as np
from torch import Tensor
from typing import Optional
from jaxtyping import Float
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from .nll import calculate_nll_normal, calculate_nll_t, calculate_nll_mol
from .ause import calculate_ause


def infer_evidential_parameters(AU, EU, alpha_minus_1, eps=1e-6):
    nu = (AU / EU).clamp(eps)
    alpha_minus_1 = alpha_minus_1.clamp(eps)
    alpha = alpha_minus_1 + 1
    beta = (AU * alpha_minus_1).clamp(eps)
    return nu, alpha, beta

def evaluate_reconstruction(
    ground_truth3: Float[Tensor, "N 3 H W"],
    prediction3: Float[Tensor, "N 3 H W"],
):
    reconstruction_functions = {
        'psnr': PeakSignalNoiseRatio(data_range=1.0),
        'ssim': structural_similarity_index_measure,
        'lpips': LearnedPerceptualImagePatchSimilarity(normalize=True),
    }
    return np.array([
        [reconstruction_functions[metric_name](ground_truth3[i:i+1], prediction3[i:i+1]).item() for metric_name in ['psnr', 'ssim', 'lpips']]
        for i in range(len(ground_truth3))
    ]).mean(0).tolist()

def evaluate_normal(
    ground_truth3: Float[Tensor, "N 3 H W"],
    prediction3: Float[Tensor, "N 3 H W"],
    uncertainty3: Optional[Float[Tensor, "N 3 H W"]] = None,
    uncertainty1: Optional[Float[Tensor, "N 1 H W"]] = None,
):
    absolute_error3 = (ground_truth3 - prediction3).abs()  # [N, 3, H, W]
    squared_error3 = absolute_error3 ** 2  # [N, 3, H, W]
    absolute_error1 = absolute_error3.mean(1)  # [N, H, W]
    squared_error1 = squared_error3.mean(1)  # [N, H, W]

    assert (uncertainty3 is None) != (uncertainty1 is None), 'Exactly one of uncertainty3 and uncertainty1 should be provided'
    if uncertainty3 is None:
        uncertainty3 = uncertainty1.expand(-1, 3, -1, -1)  # [N, 3, H, W]  # type: ignore
    elif uncertainty1 is None:
        uncertainty1 = uncertainty3.mean(1, keepdim=True)  # [N, H, W]

    nll = calculate_nll_normal(squared_error3, uncertainty3)
    ause_rmse = calculate_ause(squared_error1, uncertainty1, sqrt=True)[0]  # type: ignore
    ause_mae = calculate_ause(absolute_error1, uncertainty1, sqrt=False)[0]  # type: ignore
    return {'nll': nll, 'ause_rmse': ause_rmse, 'ause_mae': ause_mae}

def evaluate_evidential(
    ground_truth3: Float[Tensor, "N 3 H W"],
    prediction3: Float[Tensor, "N 3 H W"],
    AU1: Float[Tensor, "N 1 H W"],
    EU1: Float[Tensor, "N 1 H W"],
    alpha_minus_1: Float[Tensor, "N 1 H W"],
):
    absolute_error3 = (ground_truth3 - prediction3).abs()  # [N, 3, H, W]
    squared_error3 = absolute_error3 ** 2  # [N, 3, H, W]
    absolute_error1 = absolute_error3.mean(1, keepdim=True)  # [N, 1, H, W]
    squared_error1 = squared_error3.mean(1, keepdim=True)  # [N, 1, H, W]

    nu1, alpha1, beta1 = infer_evidential_parameters(AU1, EU1, alpha_minus_1)  # [N, 1, H, W]
    U1 = AU1 + EU1  # [N, 1, H, W]
    nu3, alpha3, beta3, U3 = [param.expand(-1, 3, -1, -1) for param in [nu1, alpha1, beta1, U1]]  # [N, 3, H, W]

    nll = calculate_nll_t(squared_error3, nu3, alpha3, beta3)
    ause_rmse = calculate_ause(squared_error1, U1, sqrt=True)[0]
    ause_mae = calculate_ause(absolute_error1, U1, sqrt=False)[0]
    return {'nll': nll, 'ause_rmse': ause_rmse, 'ause_mae': ause_mae}

def evaluate_mol(
    ground_truth3: Float[Tensor, "N 3 H W"],
    prediction3: Float[Tensor, "N 3 H W"],
    w: Float[Tensor, "N 1 H W K"],
    mu: Float[Tensor, "N 3 H W K"],
    b: Float[Tensor, "N 3 H W K"],
):
    absolute_error3 = (ground_truth3 - prediction3).abs()  # [N, 3, H, W]
    squared_error3 = absolute_error3 ** 2  # [N, 3, H, W]
    absolute_error1 = absolute_error3.mean(1, keepdim=True)  # [N, 1, H, W]
    squared_error1 = squared_error3.mean(1, keepdim=True)  # [N, 1, H, W]

    nll = calculate_nll_mol(ground_truth3, w, mu, b)

    second_moment3 = (w * (2.0 * b ** 2 + mu ** 2)).sum(-1)  # [N, 3, H, W]
    uncertainty3 = second_moment3 - prediction3 ** 2  # [N, 3, H, W]
    uncertainty1 = uncertainty3.mean(1, keepdim=True)  # [N, 1, H, W]

    ause_rmse = calculate_ause(squared_error1, uncertainty1, sqrt=True)[0]
    ause_mae = calculate_ause(absolute_error1, uncertainty1, sqrt=False)[0]

    return {'nll': nll, 'ause_rmse': ause_rmse, 'ause_mae': ause_mae}
