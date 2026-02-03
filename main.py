"""
Main Pipeline Script for Alzheimer's Detection
===============================================

This script provides a complete end-to-end pipeline for:
1. Data preprocessing
2. Model creation
3. Model training
4. Model evaluation
5. Grad-CAM visualization

Usage:
    python main.py --data_dir /path/to/dataset --output_dir ./output

Author: Final Year Project
Date: 2024
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Import project modules
from preprocessing.data_preprocessing import preprocess_pipeline
from model.cnn_model import create_alzheimer_cnn
from training.train_model import train_alzheimer_model
from evaluation.evaluate_model import evaluate_alzheimer_model
from explainability.gradcam import create_gradcam_explainer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Alzheimer\'s Detection - Complete Pipeline'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help='Input image size (square)'
    )
    
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training (use existing model)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to existing model (if skip_training)'
    )
    
    return parser.parse_args()


def setup_directories(output_dir):
    """Create necessary output directories."""
    dirs = {
        'models': os.path.join(output_dir, 'models'),
        'evaluation': os.path.join(output_dir, 'evaluation'),
        'visualizations': os.path.join(output_dir, 'visualizations'),
        'logs': os.path.join(output_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs


def save_config(args, dirs):
    """Save pipeline configuration."""
    config = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size
    }
    
    config_path = os.path.join(dirs['logs'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"[INFO] Configuration saved to: {config_path}")
    return config


def run_pipeline(args):
    """Execute complete pipeline."""
    
    print("="*70)
    print("ALZHEIMER'S DETECTION - COMPLETE PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    config = save_config(args, dirs)
    
    # Step 1: Data Preprocessing
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    preprocessor, train_gen, val_gen, class_weights = preprocess_pipeline(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Step 2 & 3: Model Creation and Training
    if not args.skip_training:
        print("\n" + "="*70)
        print("STEP 2: MODEL CREATION")
        print("="*70)
        
        model, cnn = create_alzheimer_cnn(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=4
        )
        
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING")
        print("="*70)
        
        model, trainer, history = train_alzheimer_model(
            model=model,
            train_generator=train_gen,
            validation_generator=val_gen,
            class_weights=class_weights,
            epochs=args.epochs,
            model_dir=dirs['models']
        )
        
        # Save final model
        final_model_path = os.path.join(dirs['models'], 'final_model.h5')
        model.save(final_model_path)
        print(f"[INFO] Model saved to: {final_model_path}")
        
    else:
        print("\n" + "="*70)
        print("STEP 2 & 3: SKIPPED (Using Existing Model)")
        print("="*70)
        
        if not args.model_path:
            raise ValueError("--model_path required when using --skip_training")
            
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model_path)
        print(f"[INFO] Loaded model from: {args.model_path}")
    
    # Step 4: Model Evaluation
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    metrics = evaluate_alzheimer_model(
        model=model,
        test_generator=val_gen,
        output_dir=dirs['evaluation']
    )
    
    # Step 5: Grad-CAM Visualization
    print("\n" + "="*70)
    print("STEP 5: GRAD-CAM VISUALIZATION")
    print("="*70)
    
    gradcam = create_gradcam_explainer(model)
    
    # Get sample images for visualization
    sample_images = []
    for i in range(min(5, len(val_gen))):
        batch_x, batch_y = val_gen[i]
        for j in range(min(2, len(batch_x))):
            # Save sample image temporarily
            import cv2
            import numpy as np
            img_path = os.path.join(dirs['visualizations'], f'sample_{i}_{j}.jpg')
            img = (batch_x[j] * 255).astype(np.uint8)
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            sample_images.append(img_path)
            
        if len(sample_images) >= 5:
            break
    
    # Generate Grad-CAM visualizations
    if sample_images:
        gradcam.analyze_multiple(
            image_paths=sample_images,
            preprocessor=preprocessor,
            output_dir=dirs['visualizations']
        )
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Completed at: {datetime.now()}")
    print(f"\nOutput Files:")
    print(f"  - Model: {dirs['models']}")
    print(f"  - Evaluation: {dirs['evaluation']}")
    print(f"  - Visualizations: {dirs['visualizations']}")
    print(f"  - Logs: {dirs['logs']}")
    print("="*70)
    
    return model, metrics


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        model, metrics = run_pipeline(args)
        
        print("\n✅ Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
