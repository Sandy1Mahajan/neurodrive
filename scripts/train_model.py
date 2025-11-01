"""
Train PyTorch ML model for driver risk assessment.

This script trains a neural network to predict risk scores from driver metrics.
The model can be exported to ONNX for production inference.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any
import numpy as np
import pandas as pd

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Install with: pip install torch")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TelemetryDataset(Dataset):
    """Dataset for driver telemetry."""
    
    def __init__(self, csv_path: str):
        """Load data from CSV."""
        df = pd.read_csv(csv_path)
        
        # Features: eye_closure_ratio, phone_usage, speed, head_pose_degrees, unauthorized_objects_count
        self.features = df[['eye_closure_ratio', 'phone_usage', 'speed', 
                           'head_pose_degrees', 'unauthorized_objects_count']].values
        
        # Normalize features
        self.feature_mean = np.mean(self.features, axis=0)
        self.feature_std = np.std(self.features, axis=0) + 1e-8
        self.features = (self.features - self.feature_mean) / self.feature_std
        
        # Targets: individual risk scores and overall risk
        self.targets = df[['drowsiness_risk', 'distraction_risk', 
                          'head_pose_risk', 'objects_risk', 'risk_score']].values
        
        logger.info(f"Loaded dataset: {len(self.features)} samples")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        targets = torch.FloatTensor(self.targets[idx])
        return features, targets


class RiskPredictionModel(nn.Module):
    """Neural network for risk prediction."""
    
    def __init__(self, input_dim: int = 5, hidden_dims: list = [64, 32], dropout: float = 0.2):
        """Initialize model.
        
        Args:
            input_dim: Number of input features (5: eye_closure, phone, speed, head_pose, objects)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(RiskPredictionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layers for each risk component
        self.drowsiness_head = nn.Linear(prev_dim, 1)
        self.distraction_head = nn.Linear(prev_dim, 1)
        self.head_pose_head = nn.Linear(prev_dim, 1)
        self.objects_head = nn.Linear(prev_dim, 1)
        self.overall_head = nn.Linear(prev_dim, 1)
        
        # Activation for outputs (sigmoid for 0-1 range)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
        
        Returns:
            Tuple of risk predictions
        """
        features = self.feature_extractor(x)
        
        drowsiness = self.sigmoid(self.drowsiness_head(features))
        distraction = self.sigmoid(self.distraction_head(features))
        head_pose = self.sigmoid(self.head_pose_head(features))
        objects = self.sigmoid(self.objects_head(features))
        overall = self.sigmoid(self.overall_head(features))
        
        return drowsiness, distraction, head_pose, objects, overall


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(features)
        
        # Calculate loss (MSE for each component)
        losses = []
        for pred, target in zip(preds, targets.T):
            loss = criterion(pred.squeeze(), target)
            losses.append(loss)
        
        total_loss_batch = sum(losses)
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            
            preds = model(features)
            
            losses = []
            for pred, target in zip(preds, targets.T):
                loss = criterion(pred.squeeze(), target)
                losses.append(loss)
            
            total_loss += sum(losses).item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)


def train_model(train_path: str, test_path: str, output_dir: str = "backend/models",
                epochs: int = 50, batch_size: int = 64, learning_rate: float = 0.001):
    """Train the model."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = TelemetryDataset(train_path)
    test_dataset = TelemetryDataset(test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = RiskPredictionModel(input_dim=5, hidden_dims=[64, 32], dropout=0.2)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    logger.info("Starting training...")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, "risk_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_mean': train_dataset.feature_mean,
                'feature_std': train_dataset.feature_std,
                'epoch': epoch,
                'val_loss': val_loss
            }, model_path)
            logger.info(f"Saved best model to {model_path} (val_loss: {val_loss:.4f})")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': float(best_val_loss),
        'epochs': epochs
    }
    
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info("Training complete!")
    return model, train_dataset


def export_to_onnx(model, dataset, output_path: str, device):
    """Export PyTorch model to ONNX."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.FloatTensor([[0.2, 0.0, 65.0, 5.0, 0.0]]).to(device)
    
    # Export to ONNX
    logger.info(f"Exporting model to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['features'],
        output_names=['drowsiness', 'distraction', 'head_pose', 'objects', 'overall_risk'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'drowsiness': {0: 'batch_size'},
            'distraction': {0: 'batch_size'},
            'head_pose': {0: 'batch_size'},
            'objects': {0: 'batch_size'},
            'overall_risk': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    logger.info("ONNX export complete!")
    
    # Save normalization parameters
    norm_params = {
        'feature_mean': dataset.feature_mean.tolist(),
        'feature_std': dataset.feature_std.tolist()
    }
    
    norm_path = output_path.replace('.onnx', '_norm.json')
    with open(norm_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    logger.info(f"Saved normalization parameters to {norm_path}")


def main():
    parser = argparse.ArgumentParser(description="Train driver risk prediction model")
    parser.add_argument("--train-data", type=str, default="data/train_data.csv",
                       help="Path to training CSV")
    parser.add_argument("--test-data", type=str, default="data/test_data.csv",
                       help="Path to test CSV")
    parser.add_argument("--output-dir", type=str, default="backend/models",
                       help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--export-onnx", action="store_true",
                       help="Export model to ONNX after training")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.train_data):
        logger.error(f"Training data not found: {args.train_data}")
        logger.error("Run scripts/prepare_sample_data.py first to generate data")
        sys.exit(1)
    
    # Train model
    model, dataset = train_model(
        args.train_data,
        args.test_data,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # Export to ONNX if requested
    if args.export_onnx:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        onnx_path = os.path.join(args.output_dir, "risk_model.onnx")
        export_to_onnx(model, dataset, onnx_path, device)


if __name__ == "__main__":
    main()


