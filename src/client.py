"""
Phase C2: Federated Learning Client
FlowerClient subclasses for performing local training with DP protection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import flwr as fl
from flwr.common import NDArrays, Scalar

logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.data import TBChestXrayDataset, get_train_transform, get_val_transform
from src.partition import get_client_partition
from src.dp import wrap_model_with_dp
from src.models import get_model


class FlowerClient(fl.client.NumPyClient):
    """
    Federated Learning client for local training.
    Handles data loading, model training with DP, and weight synchronization.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        learning_rate: float = config.LEARNING_RATE,
        device: str = config.DEVICE,
        dp_enabled: bool = config.DP_ENABLED,
    ):
        """
        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_dataloader: Training dataloader for this client
            val_dataloader: Validation dataloader for this client
            learning_rate: Learning rate for local SGD
            device: Device to train on ('cpu' or 'cuda')
            dp_enabled: Whether to apply differential privacy
        """
        self.client_id = client_id
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.device = device
        self.dp_enabled = dp_enabled
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup optimizer and loss
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=0.9
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Wrap with DP if enabled
        if dp_enabled:
            self.model, self.optimizer, self.dp_engine = wrap_model_with_dp(
                self.model,
                self.optimizer,
                self.train_dataloader,
                dp_enabled=True,
            )
        else:
            self.dp_engine = None
        
        # Metrics
        self.local_epochs_trained = 0
        self.training_loss_history = []
        self.validation_loss_history = []
        self.validation_acc_history = []
        
        logger.info(
            f"Client {client_id} initialized: "
            f"train_batches={len(train_dataloader)}, "
            f"val_batches={len(val_dataloader)}, "
            f"dp={dp_enabled}"
        )
    
    def get_parameters(self, config: Dict) -> NDArrays:
        """
        Extract model weights as list of numpy arrays (for transmission to server).
        
        Args:
            config: Configuration dict from server
        
        Returns:
            List of parameter arrays
        """
        # Return independent NumPy snapshots to avoid accidental shared-memory mutation.
        return [param.detach().cpu().numpy().copy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set model weights from numpy arrays received from server.
        
        Args:
            parameters: List of parameter arrays from server
        """
        with torch.no_grad():
            for param, new_param in zip(self.model.parameters(), parameters):
                # Clone input arrays so client training cannot mutate server-side NumPy buffers.
                new_tensor = torch.tensor(new_param, device=self.device)
                param.data.copy_(new_tensor)
    
    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """
        Perform local training for one round.
        
        Args:
            parameters: Model weights from server
            config: Configuration dict (may contain num_epochs, learning_rate, etc.)
        
        Returns:
            Tuple of (updated_weights, num_samples_trained, metrics_dict)
        """
        # Set weights
        self.set_parameters(parameters)
        
        # Get training config (epochs, batch size, etc.)
        num_epochs = config.get("num_epochs", config.get("local_epoch", 1))
        
        # Training loop
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_x, batch_y in self.train_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                
                # Track loss
                batch_loss = loss.item() * len(batch_y)
                epoch_loss += batch_loss
                epoch_samples += len(batch_y)
            
            epoch_loss /= max(epoch_samples, 1)
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            logger.info(
                f"Client {self.client_id}, Epoch {epoch + 1}/{num_epochs}: loss={epoch_loss:.4f}"
            )
        
        # Average loss over epochs
        avg_loss = total_loss / max(num_epochs, 1)
        self.training_loss_history.append(avg_loss)
        self.local_epochs_trained += num_epochs
        
        # Get privacy budget if DP enabled
        privacy_dict = {}
        if self.dp_engine:
            epsilon, delta = self.dp_engine.get_privacy_budget()
            privacy_dict = {"epsilon": epsilon, "delta": delta}
            logger.info(f"Client {self.client_id} privacy budget: ε={epsilon:.4f}, δ={delta}")
        
        return (
            self.get_parameters({}),  # Updated weights
            epoch_samples,  # Number of samples trained on
            {
                "loss": avg_loss,
                **privacy_dict,  # Include privacy metrics
                "local_epochs": num_epochs,
            },
        )
    
    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on validation set.
        
        Args:
            parameters: Model weights
            config: Configuration dict
        
        Returns:
            Tuple of (loss, num_samples, metrics_dict)
        """
        # Set weights
        self.set_parameters(parameters)
        
        # Evaluation loop
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in self.val_dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.loss_fn(logits, batch_y)
                
                total_loss += loss.item() * len(batch_y)
                total_correct += (logits.argmax(1) == batch_y).sum().item()
                total_samples += len(batch_y)
        
        self.model.train()
        
        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        
        self.validation_loss_history.append(avg_loss)
        self.validation_acc_history.append(accuracy)
        
        logger.info(
            f"Client {self.client_id} validation: loss={avg_loss:.4f}, acc={accuracy:.4f}"
        )
        
        return avg_loss, total_samples, {"accuracy": accuracy}


def create_client(
    client_id: int,
    train_dataset: TBChestXrayDataset,
    val_dataset: TBChestXrayDataset,
    batch_size: int = config.BATCH_SIZE,
    model_name: str = "resnet50",
    learning_rate: float = config.LEARNING_RATE,
    device: str = config.DEVICE,
    dp_enabled: bool = config.DP_ENABLED,
) -> FlowerClient:
    """
    Factory function to create a FlowerClient.
    
    Args:
        client_id: Client identifier
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for dataloaders
        model_name: Model architecture ('resnet50' or 'vgg16')
        learning_rate: Learning rate
        device: Compute device
        dp_enabled: Whether to use differential privacy
    
    Returns:
        Initialized FlowerClient
    """
    # Create model
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return FlowerClient(
        client_id=client_id,
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=learning_rate,
        device=device,
        dp_enabled=dp_enabled,
    )


if __name__ == "__main__":
    # Test client creation
    train_ds = TBChestXrayDataset(config.TB_ORGANIZED_ROOT, split="train", transform=get_train_transform())
    val_ds = TBChestXrayDataset(config.TB_ORGANIZED_ROOT, split="val", transform=get_val_transform())
    
    client = create_client(0, train_ds, val_ds)
    print(f"Client created: {client}")
