"""
Model Training Pipeline for Alzheimer's Detection
=================================================

This module handles the complete training process including:
- Model training with callbacks
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training history visualization

Training Strategy:
- Adam optimizer with learning rate decay
- Early stopping to prevent overfitting
- Model checkpointing for best weights
- Class weighting for imbalanced data

Author: Final Year Project
Date: 2024
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
import matplotlib.pyplot as plt
import json
from datetime import datetime


class ModelTrainer:
    """
    Training pipeline for Alzheimer's CNN model.
    
    Implements best practices for medical AI training:
    - Early stopping with patience
    - Learning rate reduction on plateau
    - Model checkpointing
    - Comprehensive logging
    """
    
    def __init__(self, model, model_dir='models'):
        """
        Initialize trainer with model and configuration.
        
        Args:
            model: Compiled Keras model
            model_dir: Directory to save model artifacts
        """
        self.model = model
        self.model_dir = model_dir
        self.history = None
        self.callbacks = []
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(model_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
    def setup_callbacks(self, 
                       early_stopping_patience=10,
                       reduce_lr_patience=5,
                       reduce_lr_factor=0.5,
                       min_lr=1e-7):
        """
        Setup training callbacks for optimal training.
        
        CALLBACKS EXPLAINED:
        ====================
        
        1. EARLY STOPPING:
           - Monitors validation loss
           - Stops training if no improvement for 'patience' epochs
           - Restores best weights automatically
           - Prevents overfitting and saves compute time
        
        2. MODEL CHECKPOINT:
           - Saves model at each epoch if validation loss improves
           - Ensures we keep the best model
           - Can resume training from checkpoint
        
        3. REDUCE LR ON PLATEAU:
           - Reduces learning rate when validation loss plateaus
           - Helps model converge to better minimum
           - Prevents oscillation near convergence
        
        4. TENSORBOARD:
           - Real-time visualization of training metrics
           - Compare multiple training runs
           - Analyze model behavior
        
        5. CSV LOGGER:
           - Saves epoch-wise metrics to CSV
           - Easy analysis and plotting
        
        Args:
            early_stopping_patience: Epochs to wait before stopping
            reduce_lr_patience: Epochs to wait before reducing LR
            reduce_lr_factor: Factor to multiply LR by
            min_lr: Minimum learning rate
        """
        
        # 1. Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        
        # 2. Model Checkpoint
        checkpoint_path = os.path.join(self.run_dir, 'best_model.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # 3. Learning Rate Reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        )
        
        # 4. TensorBoard
        log_dir = os.path.join(self.run_dir, 'logs')
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        
        # 5. CSV Logger
        csv_path = os.path.join(self.run_dir, 'training_history.csv')
        csv_logger = CSVLogger(csv_path, append=True)
        
        self.callbacks = [early_stopping, checkpoint, reduce_lr, tensorboard, csv_logger]
        
        print("[INFO] Callbacks configured:")
        print(f"  - Early Stopping (patience={early_stopping_patience})")
        print(f"  - Model Checkpoint: {checkpoint_path}")
        print(f"  - Reduce LR (factor={reduce_lr_factor}, patience={reduce_lr_patience})")
        print(f"  - TensorBoard: {log_dir}")
        print(f"  - CSV Logger: {csv_path}")
        
    def train(self, 
              train_generator,
              validation_generator,
              epochs=50,
              class_weights=None,
              early_stopping_patience=10):
        """
        Train the model with all configured callbacks.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Maximum number of epochs
            class_weights: Dictionary of class weights for imbalance
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history object
        """
        # Setup callbacks
        self.setup_callbacks(early_stopping_patience=early_stopping_patience)
        
        print("\n" + "="*70)
        print("STARTING MODEL TRAINING")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Batch size: {train_generator.batch_size}")
        if class_weights:
            print(f"Class weights: {class_weights}")
        
        # Calculate steps per epoch
        steps_per_epoch = train_generator.samples // train_generator.batch_size
        validation_steps = validation_generator.samples // validation_generator.batch_size
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=self.callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        
        # Save final model
        final_model_path = os.path.join(self.run_dir, 'final_model.h5')
        self.model.save(final_model_path)
        print(f"[INFO] Final model saved to: {final_model_path}")
        
        # Save training history
        self._save_history()
        
        # Plot training curves
        self._plot_training_curves()
        
        return self.history
    
    def _save_history(self):
        """Save training history to JSON file."""
        history_path = os.path.join(self.run_dir, 'history.json')
        
        # Convert numpy types to Python types for JSON serialization
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
            
        print(f"[INFO] Training history saved to: {history_path}")
    
    def _plot_training_curves(self):
        """Plot and save training/validation curves."""
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Loss
        axes[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[0, 1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Precision & Recall
        if 'precision' in history and 'val_precision' in history:
            axes[1, 0].plot(epochs, history['precision'], 'b-', label='Training Precision')
            axes[1, 0].plot(epochs, history['val_precision'], 'r-', label='Validation Precision')
            axes[1, 0].plot(epochs, history['recall'], 'g-', label='Training Recall')
            axes[1, 0].plot(epochs, history['val_recall'], 'y-', label='Validation Recall')
            axes[1, 0].set_title('Precision & Recall', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: AUC
        if 'auc' in history and 'val_auc' in history:
            axes[1, 1].plot(epochs, history['auc'], 'b-', label='Training AUC')
            axes[1, 1].plot(epochs, history['val_auc'], 'r-', label='Validation AUC')
            axes[1, 1].set_title('AUC-ROC', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Training curves saved to: {plot_path}")
    
    def get_best_metrics(self):
        """
        Get best metrics achieved during training.
        
        Returns:
            Dictionary of best metrics
        """
        if self.history is None:
            raise ValueError("Model not trained yet")
            
        history = self.history.history
        
        best_epoch = np.argmax(history['val_accuracy'])
        
        best_metrics = {
            'best_epoch': best_epoch + 1,
            'best_val_accuracy': history['val_accuracy'][best_epoch],
            'best_val_loss': history['val_loss'][best_epoch],
            'best_train_accuracy': history['accuracy'][best_epoch],
        }
        
        if 'val_precision' in history:
            best_metrics['best_val_precision'] = history['val_precision'][best_epoch]
        if 'val_recall' in history:
            best_metrics['best_val_recall'] = history['val_recall'][best_epoch]
        if 'val_auc' in history:
            best_metrics['best_val_auc'] = history['val_auc'][best_epoch]
            
        return best_metrics


def train_alzheimer_model(model, 
                          train_generator, 
                          validation_generator,
                          class_weights=None,
                          epochs=50,
                          model_dir='models'):
    """
    Complete training pipeline wrapper.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        validation_generator: Validation data generator
        class_weights: Class weights for imbalance
        epochs: Maximum epochs
        model_dir: Directory to save models
        
    Returns:
        Tuple of (trained_model, trainer, history)
    """
    print("="*70)
    print("ALZHEIMER'S DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = ModelTrainer(model, model_dir=model_dir)
    
    # Train model
    history = trainer.train(
        train_generator=train_generator,
        validation_generator=validation_generator,
        epochs=epochs,
        class_weights=class_weights,
        early_stopping_patience=10
    )
    
    # Get best metrics
    best_metrics = trainer.get_best_metrics()
    
    print("\n" + "="*70)
    print("BEST TRAINING RESULTS")
    print("="*70)
    for metric, value in best_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    return model, trainer, history


# Example usage
if __name__ == "__main__":
    print("[INFO] Training module loaded successfully")
    print("[INFO] Use train_alzheimer_model() to train your model")