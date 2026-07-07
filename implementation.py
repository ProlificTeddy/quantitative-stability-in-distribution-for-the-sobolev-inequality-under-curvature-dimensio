import numpy as np
import torch
from torch.nn.functional import normalize

def sobolev_inequality_stability(input_tensor, p=2, q=2, epsilon=1e-5):
    """
    Implements a simplified version of stability analysis for Sobolev inequality.
    This function computes the Wasserstein distance between the cumulative distribution
    of the input tensor and a reference distribution (Aubin-Talenti bubble approximation).

    Args:
        input_tensor (torch.Tensor): Input tensor representing the function.
        p (int): Norm parameter for Sobolev inequality.
        q (int): Norm parameter for the output space.
        epsilon (float): Regularization parameter for numerical stability.

    Returns:
        float: Wasserstein distance between the cumulative distributions.
    """
    # Normalize the input tensor
    input_tensor = normalize(input_tensor, p=2, dim=0)

    # Compute cumulative distribution of the input tensor
    input_cdf = torch.cumsum(input_tensor, dim=0)
    input_cdf /= input_cdf[-1]  # Normalize to make it a valid cumulative distribution

    # Generate reference distribution (Aubin-Talenti bubble approximation)
    x = torch.linspace(0, 1, input_tensor.shape[0])
    reference_distribution = (1 - x**2)**((p - 1) / (q - 1))
    reference_distribution = torch.clamp(reference_distribution, min=0)
    reference_cdf = torch.cumsum(reference_distribution, dim=0)
    reference_cdf /= reference_cdf[-1]  # Normalize to make it a valid cumulative distribution

    # Compute Wasserstein distance between the two distributions
    wasserstein_distance = torch.sum(torch.abs(input_cdf - reference_cdf))

    return wasserstein_distance.item()

if __name__ == '__main__':
    # Generate dummy data
    dummy_data = torch.rand(100)
    
    # Test the Sobolev inequality stability function
    distance = sobolev_inequality_stability(dummy_data)
    print(f"Wasserstein distance: {distance}")