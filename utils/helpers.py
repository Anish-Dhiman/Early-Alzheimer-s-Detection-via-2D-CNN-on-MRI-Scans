"""
Utility Helper Functions for Alzheimer's Detection Project
==========================================================

Common utility functions used across the project.

Author: Final Year Project
Date: 2024
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def save_config(config, filepath='config.json'):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Configuration saved to: {filepath}")


def load_config(filepath='config.json'):
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"[INFO] Configuration loaded from: {filepath}")
    return config


def create_experiment_dir(base_dir='experiments'):
    """
    Create timestamped experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to created experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[INFO] Experiment directory created: {exp_dir}")
    return exp_dir


def plot_sample_images(generator, class_names, num_samples=8, save_path=None):
    """
    Plot sample images from data generator.
    
    Args:
        generator: Keras data generator
        class_names: List of class names
        num_samples: Number of samples to plot
        save_path: Path to save plot
    """
    # Get a batch of images
    images, labels = next(generator)
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        class_idx = np.argmax(labels[i])
        axes[i].set_title(class_names[class_idx])
        axes[i].axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Sample images saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()


def print_dataset_statistics(generator, class_names):
    """
    Print dataset statistics.
    
    Args:
        generator: Keras data generator
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total samples: {generator.samples}")
    print(f"Batch size: {generator.batch_size}")
    print(f"Steps per epoch: {generator.samples // generator.batch_size}")
    print(f"Image shape: {generator.image_shape}")
    print("\nClass distribution:")
    
    # Count samples per class
    class_counts = {}
    for i in range(len(generator)):
        _, labels = generator[i]
        for label in labels:
            class_idx = np.argmax(label)
            class_name = class_names[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / generator.samples) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")


def get_model_size(model):
    """
    Get model size in MB.
    
    Args:
        model: Keras model
        
    Returns:
        Model size in MB
    """
    # Save model to temp file
    temp_path = 'temp_model.h5'
    model.save(temp_path)
    
    # Get size
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    
    # Remove temp file
    os.remove(temp_path)
    
    return size_mb


def log_training_start(config, log_file='training.log'):
    """
    Log training start with configuration.
    
    Args:
        config: Training configuration
        log_file: Path to log file
    """
    with open(log_file, 'a') as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Training started at: {datetime.now()}\n")
        f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
        f.write("="*60 + "\n")


def log_message(message, log_file='training.log'):
    """
    Log a message to file.
    
    Args:
        message: Message to log
        log_file: Path to log file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")


class TrainingLogger:
    """
    Training logger for tracking metrics during training.
    """
    
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {}
        
    def log_epoch(self, epoch, logs):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            logs: Dictionary of metrics
        """
        for metric, value in logs.items():
            if metric not in self.metrics:
                self.metrics[metric] = []
            self.metrics[metric].append(value)
            
        # Save to file
        log_file = os.path.join(self.log_dir, 'metrics.json')
        with open(log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def plot_metrics(self, save_path=None):
        """
        Plot logged metrics.
        
        Args:
            save_path: Path to save plot
        """
        if not self.metrics:
            print("[WARNING] No metrics to plot")
            return
            
        num_metrics = len(self.metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 4))
        
        if num_metrics == 1:
            axes = [axes]
            
        for ax, (metric, values) in zip(axes, self.metrics.items()):
            ax.plot(values)
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()


# Example usage
if __name__ == "__main__":
    print("[INFO] Utilities module loaded successfully")
