"""
Generate synthetic telemetry and sample face keypoints/metrics for training and testing.

This script creates realistic synthetic data for:
- Training ML models
- End-to-end testing
- Model validation
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_telemetry(num_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic telemetry data with realistic distributions.
    
    Args:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic telemetry
    """
    np.random.seed(seed)
    
    data = []
    
    for i in range(num_samples):
        # Simulate different driving scenarios
        scenario = np.random.choice(['normal', 'drowsy', 'distracted', 'dangerous'], 
                                   p=[0.7, 0.15, 0.1, 0.05])
        
        if scenario == 'normal':
            eye_closure_ratio = np.random.beta(2, 8)  # Mostly open
            phone_usage = False
            speed = np.random.normal(65, 10)  # Normal speed
            speed = max(40, min(90, speed))
            head_pose_degrees = np.random.normal(0, 5)  # Mostly forward
            unauthorized_objects_count = 0 if np.random.random() > 0.05 else 1
        
        elif scenario == 'drowsy':
            eye_closure_ratio = np.random.beta(5, 3)  # Often closed
            phone_usage = False
            speed = np.random.normal(55, 8)  # Slower
            speed = max(40, min(85, speed))
            head_pose_degrees = np.random.normal(-5, 10)  # Slight downward
            unauthorized_objects_count = 0
        
        elif scenario == 'distracted':
            eye_closure_ratio = np.random.beta(2, 8)  # Eyes open
            phone_usage = True
            speed = np.random.normal(70, 15)  # Variable speed
            speed = max(40, min(100, speed))
            head_pose_degrees = np.random.normal(25, 15)  # Looking away
            unauthorized_objects_count = 1 if np.random.random() > 0.5 else 0
        
        else:  # dangerous
            eye_closure_ratio = np.random.beta(6, 2)  # Very drowsy
            phone_usage = np.random.random() > 0.3
            speed = np.random.normal(95, 20)  # High speed
            speed = max(60, min(140, speed))
            head_pose_degrees = np.random.normal(30, 20)  # Significant deviation
            unauthorized_objects_count = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        
        # Calculate ground truth risk score (0-1, higher is worse)
        eye_closure_time = eye_closure_ratio * 3.0
        drowsiness_risk = min(1.0, max(0.0, (eye_closure_time - 1.0) / 2.0))
        distraction_risk = 0.8 if phone_usage else 0.1
        head_pose_risk = min(1.0, abs(head_pose_degrees) / 45.0)
        objects_risk = min(1.0, unauthorized_objects_count / 2.0)
        speed_risk = min(1.0, max(0.0, (speed - 60) / 40.0))
        
        # Weighted combination
        weights = [0.25, 0.30, 0.25, 0.20]
        risks = [drowsiness_risk, distraction_risk, head_pose_risk, objects_risk]
        base_risk = sum(w * r for w, r in zip(weights, risks))
        overall_risk = base_risk * (0.7 + 0.3 * speed_risk)  # Speed multiplier
        
        # Clamp and convert to 0-100 scale (inverted: 100 = safe, 0 = dangerous)
        risk_score_100 = int(round((1.0 - overall_risk) * 100))
        
        data.append({
            'eye_closure_ratio': float(eye_closure_ratio),
            'phone_usage': int(phone_usage),
            'speed': int(round(speed)),
            'head_pose_degrees': float(abs(head_pose_degrees)),
            'unauthorized_objects_count': int(unauthorized_objects_count),
            'scenario': scenario,
            'risk_score': float(overall_risk),
            'risk_score_100': risk_score_100,
            'drowsiness_risk': float(drowsiness_risk),
            'distraction_risk': float(distraction_risk),
            'head_pose_risk': float(head_pose_risk),
            'objects_risk': float(objects_risk),
            'speed_risk': float(speed_risk),
            'timestamp': datetime.now().isoformat()
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {num_samples} synthetic telemetry samples")
    logger.info(f"Risk score distribution: min={df['risk_score_100'].min()}, "
                f"max={df['risk_score_100'].max()}, mean={df['risk_score_100'].mean():.2f}")
    
    return df


def generate_face_keypoints(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate synthetic facial landmark data for testing.
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        List of dictionaries with face keypoints
    """
    np.random.seed(42)
    
    keypoints = []
    
    for i in range(num_samples):
        # Generate 68 facial landmarks (standard format)
        # Simplified: generate key face regions
        face_width = np.random.normal(200, 20)
        face_height = np.random.normal(250, 25)
        
        # Left eye landmarks (6 points)
        left_eye_center_x = face_width * 0.35
        left_eye_center_y = face_height * 0.4
        left_eye = [
            (int(left_eye_center_x + np.random.normal(0, 10)), 
             int(left_eye_center_y + np.random.normal(0, 5)))
            for _ in range(6)
        ]
        
        # Right eye landmarks (6 points)
        right_eye_center_x = face_width * 0.65
        right_eye_center_y = face_height * 0.4
        right_eye = [
            (int(right_eye_center_x + np.random.normal(0, 10)),
             int(right_eye_center_y + np.random.normal(0, 5)))
            for _ in range(6)
        ]
        
        # Nose landmarks (9 points)
        nose_center_x = face_width * 0.5
        nose_center_y = face_height * 0.55
        nose = [
            (int(nose_center_x + np.random.normal(0, 8)),
             int(nose_center_y + np.random.normal(0, 8)))
            for _ in range(9)
        ]
        
        # Mouth landmarks (20 points)
        mouth_center_x = face_width * 0.5
        mouth_center_y = face_height * 0.7
        mouth = [
            (int(mouth_center_x + np.random.normal(0, 15)),
             int(mouth_center_y + np.random.normal(0, 8)))
            for _ in range(20)
        ]
        
        # Combine into full face landmarks (68 points total)
        face_landmarks = left_eye + right_eye + nose + mouth + [
            (int(face_width * x), int(face_height * y))
            for x, y in [(0.2, 0.3), (0.8, 0.3), (0.2, 0.9), (0.8, 0.9)]  # Face boundary
        ]
        
        # Ensure exactly 68 points
        while len(face_landmarks) < 68:
            face_landmarks.append((int(face_width * 0.5), int(face_height * 0.5)))
        face_landmarks = face_landmarks[:68]
        
        # Calculate EAR (Eye Aspect Ratio)
        def calculate_ear(eye_points):
            if len(eye_points) < 6:
                return 0.3
            # Simplified EAR calculation
            vertical_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            vertical_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            if horizontal == 0:
                return 0.3
            return (vertical_1 + vertical_2) / (2.0 * horizontal)
        
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        keypoints.append({
            'sample_id': i,
            'face_landmarks': face_landmarks,
            'left_eye_landmarks': left_eye,
            'right_eye_landmarks': right_eye,
            'nose_landmarks': nose,
            'mouth_landmarks': mouth,
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            'avg_ear': float(avg_ear),
            'face_width': int(face_width),
            'face_height': int(face_height)
        })
    
    logger.info(f"Generated {num_samples} face keypoint samples")
    return keypoints


def save_datasets(df_telemetry: pd.DataFrame, face_keypoints: List[Dict[str, Any]], 
                 output_dir: str = "data"):
    """Save generated datasets to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save telemetry as CSV and JSON
    csv_path = os.path.join(output_dir, "synthetic_telemetry.csv")
    json_path = os.path.join(output_dir, "synthetic_telemetry.json")
    
    df_telemetry.to_csv(csv_path, index=False)
    logger.info(f"Saved telemetry CSV to {csv_path}")
    
    with open(json_path, 'w') as f:
        json.dump(df_telemetry.to_dict('records'), f, indent=2)
    logger.info(f"Saved telemetry JSON to {json_path}")
    
    # Save face keypoints
    keypoints_path = os.path.join(output_dir, "synthetic_face_keypoints.json")
    with open(keypoints_path, 'w') as f:
        json.dump(face_keypoints, f, indent=2)
    logger.info(f"Saved face keypoints to {keypoints_path}")
    
    # Save training split (80% train, 20% test)
    train_size = int(len(df_telemetry) * 0.8)
    df_train = df_telemetry.iloc[:train_size]
    df_test = df_telemetry.iloc[train_size:]
    
    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    logger.info(f"Saved training split: train={len(df_train)}, test={len(df_test)}")
    
    # Save metadata
    metadata = {
        'num_samples': len(df_telemetry),
        'train_size': len(df_train),
        'test_size': len(df_test),
        'num_keypoints': len(face_keypoints),
        'generated_at': datetime.now().isoformat(),
        'columns': list(df_telemetry.columns),
        'statistics': {
            'risk_score_100': {
                'min': float(df_telemetry['risk_score_100'].min()),
                'max': float(df_telemetry['risk_score_100'].max()),
                'mean': float(df_telemetry['risk_score_100'].mean()),
                'std': float(df_telemetry['risk_score_100'].std())
            }
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic telemetry data")
    parser.add_argument("--num-telemetry", type=int, default=10000,
                       help="Number of telemetry samples to generate")
    parser.add_argument("--num-keypoints", type=int, default=1000,
                       help="Number of face keypoint samples to generate")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for datasets")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    logger.info("Generating synthetic datasets...")
    
    # Generate datasets
    df_telemetry = generate_synthetic_telemetry(args.num_telemetry, args.seed)
    face_keypoints = generate_face_keypoints(args.num_keypoints)
    
    # Save datasets
    save_datasets(df_telemetry, face_keypoints, args.output_dir)
    
    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()


