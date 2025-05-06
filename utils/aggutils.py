import torch
import torch.nn.functional as F 

from typing import Callable


def frechet_mean(vectors: torch.Tensor, distance_fn: Callable[[torch.Tensor, torch.Tensor], float], tol=1e-6, max_iter=100):
    """
    Compute the Fréchet mean of a set of vectors using a given distance function.

    Parameters:
    - vectors (torch.Tensor): A (n x d) tensor where each row is a d-dimensional vector.
    - distance_fn (callable): A function that computes the distance between two vectors.
    - tol (float): Convergence tolerance.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - torch.Tensor: The Fréchet mean (a d-dimensional tensor).
    """

    # Initialize mean as the Euclidean mean (good starting point)
    mean = vectors.mean(dim=0)

    for _ in range(max_iter):
        # Compute distances from the current mean to all vectors
        distances = torch.tensor([distance_fn(mean, v) for v in vectors])

        # Compute weights based on inverse distances (avoid division by zero)
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()  # Normalize weights to sum to 1

        # Compute new mean as weighted sum of vectors
        new_mean = torch.sum(weights[:, None] * vectors, dim=0)

        # Check for convergence
        if torch.norm(new_mean - mean) < tol:
            break

        mean = new_mean

    return mean


def marginal_median(x):
    return x.median(dim = 0).values


def variance(x):
    """Compute variance across rows (per feature)."""
    mean = x.mean(dim=0, keepdim=True)
    return ((x - mean) ** 2).mean(dim=0)


def skewness(x):
    """Compute skewness across rows (per feature)."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    return (((x - mean) / std) ** 3).mean(dim=0)


def kurtosis(x):
    """Compute excess kurtosis (Fisher definition) across rows (per feature)."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True)
    return (((x - mean) / std) ** 4).mean(dim=0) - 3  # Subtract 3 for excess kurtosis


def row_sum(x):
    """Compute the sum of each row."""
    return x.sum(dim=0)


def row_max(x):
    """Compute the maximum value of each row."""
    return x.max(dim=0).values


def row_min(x):
    """Compute the minimum value of each row."""
    return x.min(dim=0).values


def norm_max(x):
    """Return the row(s) with the greatest Euclidean norm."""
    norms = torch.norm(x, dim=1)  # Compute the L2 norm of each row
    max_norm_idx = torch.argmax(norms)  # Find the index of the max norm
    return x[max_norm_idx]

def norm_min(x):
    """Return the row(s) with the smallest Euclidean norm."""
    norms = torch.norm(x, dim=1)  # Compute the L2 norm of each row
    min_norm_idx = torch.argmin(norms)  # Find the index of the min norm
    return x[min_norm_idx]


def log_sum_exp(x):
    """Compute log-sum-exp across rows (numerically stable)."""
    max_vals, _ = x.max(dim=0, keepdim=True)  # Maximum per row for numerical stability
    return max_vals.squeeze() + torch.log(torch.sum(torch.exp(x - max_vals), dim=0))


def medoid(x, p = 2):
    """
    Compute the medoid of a set of vectors using torch.cdist.

    Args:
        x (torch.Tensor): A tensor of shape (N, D), where N is the number of rows (vectors), and D is the feature dimension.
        p (int): The p value for the p-norm distance to calculate between each vector pair.

    Returns:
        torch.Tensor: The medoid row (vector).
    """    
    # Compute pairwise distance matrix using cdist
    distance_matrix = torch.cdist(x, x, p = p)
    
    # Sum distances for each row
    total_distances = distance_matrix.sum(dim=1)

    # Find the index of the row with the smallest total distance
    medoid_index = torch.argmin(total_distances)
    
    return x[medoid_index]


def midpoint(points):
    """
    Compute the diameter of the point cloud and return the point that lies between
    the two most distant points (i.e., the midpoint of the two most distant points).
    
    Parameters:
    - points: A torch tensor of shape (n_points, n_features), where each row represents a point in the point cloud.
    
    Returns:
    - The midpoint of the two most distant points (torch tensor of shape (n_features,)).
    """
    # Compute pairwise squared Euclidean distances between all points
    diff = points.unsqueeze(0) - points.unsqueeze(1)
    distances_squared = torch.sum(diff**2, dim=2)
    
    # Compute the pairwise distances (sqrt of squared distances)
    distances = torch.sqrt(distances_squared)
    
    # Find the indices of the two most distant points
    i, j = torch.unravel_index(torch.argmax(distances), distances.shape)
    
    # Get the two most distant points
    point1 = points[i]
    point2 = points[j]
    
    # Compute the midpoint of the two points
    mid_point = (point1 + point2) / 2
    
    return mid_point


def L1_medoid(x):
    return medoid(x, p = 1)


def L2_medoid(x):
    return medoid(x, p = 2)


def P1_mean(x):
    return torch.mean(x, dim = 0)


def P2_mean(x):
    return torch.pow(torch.mean(torch.pow(x, 2), dim=0, keepdim=True), 0.5)    


def riemannian_mean(x):
    # Define cosine distance
    cosine_distance = lambda x, y: 1. - F.cosine_similarity(x, y, dim = 0)
    return frechet_mean(x, distance_fn=cosine_distance)


def geometric_median(x):
    # Define L1 distance
    L1_distance = lambda x, y: torch.sum(torch.abs(x - y))
    return frechet_mean(x, distance_fn=L1_distance)


def get_aggregation_function(function_name):
    """
    Return a function based on its name.

    Args:
        function_name (str): The name of the function to retrieve.

    Returns:
        function: The corresponding function.
    """
    # Check if the function exists in the global scope and return it
    if function_name in globals():
        return globals()[function_name]
    else:
        raise ValueError(f"Function {function_name} not found.")    