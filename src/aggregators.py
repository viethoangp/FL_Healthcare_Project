"""
Phase C4: Aggregation Functions
FedAvg (parameter averaging) and FedSGD (gradient-based) with divergence metric.
"""

from typing import List, Tuple, Callable
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_divergence(
    client_params_list: List[np.ndarray],
    global_params: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute divergence metric δ_t between client models and global model.
    Per paper: δ_t = (1/K) * sum(||w_t^k - w_{t-1}||_2^2)

    We compute the average squared L2 distance. Optionally normalize by the global
    weight norm if DIVERGENCE_NORMALIZE is enabled. You can also reduce by the
    number of parameters (mean) to keep scale aligned with tau.
    
    Args:
        client_params_list: List of K client parameter lists (each from get_parameters())
        global_params: Current global model parameters
    
    Returns:
        Divergence metric (float), normalized by ||w_global||_2
    """
    if not client_params_list:
        logger.warning("compute_divergence: empty client_params_list")
        return 0.0
    
    num_clients = len(client_params_list)
    total_distance = 0.0

    if config.DIVERGENCE_NORMALIZE:
        global_sq = 0.0
        for global_param in global_params:
            global_sq += np.sum(global_param ** 2)
        global_norm = np.sqrt(global_sq) + eps
    else:
        global_norm = 1.0

    if config.DIVERGENCE_SCOPE == "head":
        global_params = global_params[-2:]
        client_params_list = [params[-2:] for params in client_params_list]

    reduce_mode = config.DIVERGENCE_REDUCTION
    
    for client_params in client_params_list:
        # Compute squared L2 distance between client and global weights
        distance = 0.0
        for client_param, global_param in zip(client_params, global_params):
            # Flatten to 1D and compute L2 norm
            diff = (client_param - global_param).flatten()
            distance += np.sum(diff ** 2)

        if reduce_mode == "mean":
            param_count = 0
            for global_param in global_params:
                param_count += global_param.size
            param_count = max(param_count, 1)
            distance = distance / param_count

        total_distance += distance / global_norm
    
    divergence = total_distance / num_clients
    return divergence


def recommend_tau(
    divergence_history: List[float],
    quantile: float = 0.5,
) -> float:
    """
    Recommend τ based on a quantile of observed divergence values.

    Args:
        divergence_history: List of divergence values from previous rounds
        quantile: Quantile to use for recommendation (0.0-1.0)

    Returns:
        Suggested τ value
    """
    if not divergence_history:
        return 0.0

    values = np.array(divergence_history, dtype=np.float64)
    return float(np.quantile(values, quantile))


def aggregete_fedavg(
    client_params_list: List[List[np.ndarray]],
    client_sample_counts: List[int],
) -> List[np.ndarray]:
    """
    FedAvg aggregation: weighted average of client parameters.
    
    Args:
        client_params_list: List of parameter lists from K clients
        client_sample_counts: Number of local samples each client trained on
    
    Returns:
        Aggregated global parameters
    """
    if not client_params_list:
        raise ValueError("No clients to aggregate")
    
    num_clients = len(client_params_list)
    total_samples = sum(client_sample_counts)
    
    # Weighted average
    aggregated = None
    for client_idx, client_params in enumerate(client_params_list):
        weight = client_sample_counts[client_idx] / total_samples
        
        if aggregated is None:
            aggregated = [param * weight for param in client_params]
        else:
            for i, param in enumerate(client_params):
                aggregated[i] += param * weight
    
    logger.debug(f"FedAvg: aggregated {num_clients} clients, total_samples={total_samples}")
    return aggregated


def aggregate_fedsgd(
    client_params_list: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    learning_rate: float,
    client_sample_counts: List[int],
) -> List[np.ndarray]:
    """
        FedSGD aggregation: server-side gradient-style update from client deltas.
    
        We only receive updated client weights (not raw gradients), so we approximate
        server gradient direction using parameter deltas:
            delta_k = w_k - w_global
            w_next = w_global + lr_server * sum_k(p_k * delta_k)
    
    Args:
        client_params_list: List of updated parameter lists from K clients
        global_params: Current global parameters
        learning_rate: Server-side learning rate
        client_sample_counts: Number of samples each client trained on
    
    Returns:
        Updated global parameters
    """
    if not client_params_list:
        raise ValueError("No clients to aggregate")
    
    num_clients = len(client_params_list)
    total_samples = sum(client_sample_counts)
    
    # Start from current global parameters and apply weighted delta step.
    aggregated = [np.copy(param) for param in global_params]
    
    for client_idx, client_params in enumerate(client_params_list):
        weight = client_sample_counts[client_idx] / total_samples
        
        for i, (client_param, global_param) in enumerate(
            zip(client_params, global_params)
        ):
            # Move toward client updates (do not invert update direction).
            update = client_param - global_param
            aggregated[i] += (weight * learning_rate * update)
    
    logger.debug(f"FedSGD: aggregated {num_clients} clients with lr={learning_rate}")
    return aggregated


def aggregate_adaptive(
    client_params_list: List[List[np.ndarray]],
    global_params: List[np.ndarray],
    tau: float,
    learning_rate: float,
    client_sample_counts: List[int],
) -> Tuple[List[np.ndarray], str, float]:
    """
    Adaptive aggregation: choose FedAvg or FedSGD based on divergence threshold.
    
    Per paper:
    - If δ_t > τ: Use FedSGD (clients are diverging, need more aggressive update)
    - If δ_t ≤ τ: Use FedAvg (clients converging nicely, just average)
    
    Args:
        client_params_list: List of client parameter updates
        global_params: Current global parameters
        tau: Divergence threshold for switching
        learning_rate: Learning rate for FedSGD if used
        client_sample_counts: Number of samples each client trained on
    
    Returns:
        Tuple of (aggregated_params, algorithm_used, divergence_metric)
    """
    # Compute divergence
    divergence = compute_divergence(client_params_list, global_params)
    
    # Choose algorithm based on threshold
    if divergence > tau:
        aggregated = aggregate_fedsgd(
            client_params_list, global_params, learning_rate, client_sample_counts
        )
        algorithm = "FedSGD"
        logger.info(
            f"Adaptive aggregation: divergence={divergence:.6f} > τ={tau}, using {algorithm}"
        )
    else:
        aggregated = aggregete_fedavg(client_params_list, client_sample_counts)
        algorithm = "FedAvg"
        logger.info(
            f"Adaptive aggregation: divergence={divergence:.6f} ≤ τ={tau}, using {algorithm}"
        )
    
    return aggregated, algorithm, divergence


if __name__ == "__main__":
    # Test aggregation functions
    # Simulate 3 clients with some parameters
    
    num_params = 100
    num_clients = 3
    
    global_params = [np.random.randn(10, 10) for _ in range(num_params // 10)]
    client_params = [
        [np.random.randn(10, 10) + 0.1 * np.random.randn(10, 10) for _ in range(num_params // 10)]
        for _ in range(num_clients)
    ]
    sample_counts = [100, 120, 150]
    
    # Test divergence
    div = compute_divergence(client_params, global_params)
    print(f"Divergence: {div:.6f}")
    
    # Test FedAvg
    aggregated_avg = aggregete_fedavg(client_params, sample_counts)
    print(f"FedAvg aggregation complete, num_params={len(aggregated_avg)}")
    
    # Test FedSGD
    aggregated_sgd = aggregate_fedsgd(client_params, global_params, lr=0.01, client_sample_counts=sample_counts)
    print(f"FedSGD aggregation complete, num_params={len(aggregated_sgd)}")
    
    # Test adaptive
    agg, algo, div = aggregate_adaptive(client_params, global_params, tau=0.1, learning_rate=0.01, client_sample_counts=sample_counts)
    print(f"Adaptive aggregation: algorithm={algo}, divergence={div:.6f}")
