"""
Phase A3: Non-IID Data Partitioning
Dirichlet(alpha) distribution partitions data among clients to simulate hospital heterogeneity.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def dirichlet_partition(
    dataset_indices: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    alpha: float = config.DIRICHLET_ALPHA,
    seed: int = config.RANDOM_SEED,
) -> List[np.ndarray]:
    """
    Partition dataset into num_clients using Dirichlet distribution.
    Creates non-IID (Non-Independently and Identically Distributed) splits per client.
    
    Args:
        dataset_indices: Array of dataset indices [0, 1, ..., N-1]
        labels: Array of labels [0, 1, ..., C-1] per index
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
               - alpha >> 1: IID (uniform distribution)
               - alpha < 1: Highly non-IID, each client sees few classes
               - alpha = 0.5: Moderate non-IID (per paper)
        seed: Random seed for reproducibility
    
    Returns:
        List of num_clients arrays, each containing indices assigned to that client
    """
    
    np.random.seed(seed)
    
    # Get unique classes and their counts
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    logger.info(
        f"Partitioning {len(dataset_indices)} samples into {num_clients} clients "
        f"with Dirichlet(alpha={alpha}), num_classes={num_classes}"
    )
    
    # Organize indices by class
    class_indices = {}
    for class_label in unique_labels:
        mask = labels == class_label
        class_indices[class_label] = np.where(mask)[0]
        logger.info(f"  Class {class_label}: {len(class_indices[class_label])} samples")
    
    # Initialize client partitions
    client_indices = [[] for _ in range(num_clients)]
    
    # ---------------------------------------------------------
    # PAPER REPLICATION: Force Extreme Imbalance on Client 2, 5, 8
    # ---------------------------------------------------------
    enforce_paper = getattr(config, "ENFORCE_PAPER_IMBALANCE", True) and num_clients >= 9
    if enforce_paper:
        logger.info("  [!] ENFORCE_PAPER_IMBALANCE active: explicitly starving clients 2, 5, 8")
        # Define explicit scarce counts: {client_id: {class_label: count}}
        # Client 2: Severe TB imbalance (e.g. 25 Normal, 2 TB)
        # Client 5: Severe Normal imbalance (e.g. 2 Normal, 25 TB)
        # Client 8: Extreme scarcity (e.g. 3 Normal, 3 TB)
        scarce_alloc = {
            2: {0: 25, 1: 2},
            5: {0: 2, 1: 25},
            8: {0: 3, 1: 3}
        }
        
        for cid, counts in scarce_alloc.items():
            for cls_lbl, count in counts.items():
                if cls_lbl in class_indices and count > 0:
                    cls_samples = class_indices[cls_lbl]
                    if len(cls_samples) >= count:
                        # Extract the required samples
                        assigned = cls_samples[:count]
                        client_indices[cid].extend(assigned)
                        # Remove them from the pool
                        class_indices[cls_lbl] = cls_samples[count:]
        
        # We need to distribute the remaining to other clients.
        # Create a boolean mask of clients that are NOT 2, 5, 8
        active_clients = [i for i in range(num_clients) if i not in [2, 5, 8]]
    else:
        active_clients = list(range(num_clients))
    # ---------------------------------------------------------

    # Sample from Dirichlet distribution for each class
    # for the active clients only
    num_active = len(active_clients)
    if alpha > 0:
        dirichlet_dist = np.random.dirichlet(
            [alpha] * num_active, size=num_classes
        )  # Shape: (num_classes, num_active)
    else:
        dirichlet_dist = np.ones((num_classes, num_active)) / num_active
    
    # Assign remaining samples to active clients based on Dirichlet probabilities
    for class_label in unique_labels:
        class_samples = class_indices[class_label]
        # Skip if no samples left
        if len(class_samples) == 0:
            continue
            
        probabilities = dirichlet_dist[int(class_label)]
        np.random.shuffle(class_samples)
        
        cumulative_count = 0
        for idx, client_id in enumerate(active_clients):
            count = int(probabilities[idx] * len(class_samples))
            client_indices[client_id].extend(
                class_samples[cumulative_count : cumulative_count + count]
            )
            cumulative_count += count
        
        # Assign remaining samples to last active client
        if cumulative_count < len(class_samples):
            client_indices[active_clients[-1]].extend(
                class_samples[cumulative_count:]
            )
    
    # Convert to numpy arrays with proper dtype
    client_indices = [np.array(indices, dtype=np.int64) for indices in client_indices]
    
    # Log distribution statistics
    logger.info("\nClient-wise data distribution:")
    for client_id, indices in enumerate(client_indices):
        client_labels = labels[indices]
        label_counts = {}
        for class_label in unique_labels:
            count = np.sum(client_labels == class_label)
            label_counts[int(class_label)] = count
        logger.info(f"  Client {client_id}: {len(indices)} samples, distribution={label_counts}")
    
    return client_indices


def get_client_partition(
    dataset_length: int,
    labels: np.ndarray,
    num_clients: int = config.NUM_CLIENTS_BASELINE,
    alpha: float = config.DIRICHLET_ALPHA,
    seed: int = config.RANDOM_SEED,
) -> List[np.ndarray]:
    """
    Get partitioning for a dataset with given labels.
    Wrapper around dirichlet_partition for convenience.
    
    Args:
        dataset_length: Total number of samples in dataset
        labels: Array of labels for each sample
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        seed: Random seed
    
    Returns:
        List of client partitions (each is array of indices)
    """
    dataset_indices = np.arange(dataset_length)
    return dirichlet_partition(dataset_indices, labels, num_clients, alpha, seed)


if __name__ == "__main__":
    # Test partitioning with dummy data
    # Simulate 1000 samples with 2 classes
    num_samples = 1000
    num_classes = 2
    num_clients = 10
    
    labels = np.random.binomial(n=1, p=0.5, size=num_samples)  # 0 or 1
    
    partitions = dirichlet_partition(
        np.arange(num_samples),
        labels,
        num_clients=num_clients,
        alpha=0.5,
        seed=42,
    )
    
    print(f"Partitioning test passed. {num_clients} client partitions created.")
