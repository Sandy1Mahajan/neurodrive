"""
Export PyTorch model to ONNX format for production inference.

This script loads a trained PyTorch model and exports it to ONNX format.
"""

import os
import sys
import argparse
import torch
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.train_model import RiskPredictionModel, TelemetryDataset

def load_pytorch_model(model_path: str, device: str = "cpu"):
    """Load trained PyTorch model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = RiskPredictionModel(input_dim=5, hidden_dims=[64, 32], dropout=0.2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model, checkpoint


def export_to_onnx(model, output_path: str, device: str = "cpu"):
    """Export model to ONNX."""
    model = model.to(device)
    model.eval()
    
    # Create dummy input (normalized features)
    dummy_input = torch.FloatTensor([[0.2, 0.0, 65.0, 5.0, 0.0]]).to(device)
    
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    try:
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
            do_constant_folding=True,
            export_params=True
        )
        logger.info(f"Successfully exported ONNX model to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model-path", type=str, default="backend/models/risk_model.pth",
                       help="Path to PyTorch model checkpoint")
    parser.add_argument("--output", type=str, default="backend/models/risk_model.onnx",
                       help="Output ONNX file path")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        logger.error("Train model first with: python scripts/train_model.py")
        sys.exit(1)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    # Load model
    model, checkpoint = load_pytorch_model(args.model_path, device)
    
    # Export to ONNX
    success = export_to_onnx(model, args.output, device)
    
    if success:
        logger.info("ONNX export complete!")
        logger.info(f"Model saved to: {args.output}")
        logger.info("\nTo use in production, set MODEL_TYPE=onnx and MODEL_PATH in environment variables")
    else:
        logger.error("ONNX export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()


