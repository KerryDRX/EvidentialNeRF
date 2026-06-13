from typing import Union
import torch
from torch import Tensor
from jaxtyping import Float


def calculate_nll_normal(
    squared_error3: Float[Tensor, "N 3 H W"],
    uncertainty3: Float[Tensor, "N 3 H W"],
):
    return ((squared_error3 / uncertainty3 + (2 * torch.pi * uncertainty3).log()) / 2).mean().item()

def calculate_nll_t(
    squared_error3: Float[Tensor, "N 3 H W"],
    nu3: Float[Tensor, "N 3 H W"],
    alpha3: Float[Tensor, "N 3 H W"],
    beta3: Float[Tensor, "N 3 H W"],
):
    omega = 2 * beta3 * (1 + nu3)
    return (
        (torch.pi / nu3).log() / 2
        - alpha3 * omega.log()
        + alpha3.lgamma() - (alpha3 + 0.5).lgamma()
        + (alpha3 + 0.5) * (squared_error3 * nu3 + omega).log()
    ).mean().item()

def calculate_nll_mol(
    ground_truth3: Float[Tensor, "N 3 H W"],
    w: Float[Tensor, "N 1 H W K"],
    mu: Float[Tensor, "N 3 H W K"],
    b: Float[Tensor, "N 3 H W K"],
    as_item: bool = True,
) -> Union[float, Tensor]:
    log_pdf = -(ground_truth3.unsqueeze(-1) - mu).abs() / b - (2.0 * b).log()  # [N, 3, H, W, K]
    log_w = (w + 1e-12).log()  # [N, 1, H, W, K]
    log_likelihood = torch.logsumexp(log_w + log_pdf, dim=-1)  # [N, 3, H, W]
    nll = -log_likelihood.mean()
    return nll.item() if as_item else nll
