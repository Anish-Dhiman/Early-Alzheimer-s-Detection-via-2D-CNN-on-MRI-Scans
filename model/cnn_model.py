"""
Custom 2D CNN Architecture for Alzheimer's Detection
=====================================================

This module implements a custom 2D Convolutional Neural Network
specifically designed for multi-class classification of Alzheimer's
disease stages from MRI scan images.

WHY 2D CNN INSTEAD OF 3D CNN?
==============================
1. COMPUTATIONAL EFFICIENCY:
   - 2D CNNs require significantly less memory and compute
   - Can train on standard GPUs (8-16 GB VRAM)
   - Faster training and inference times

2. DATA AVAILABILITY:
   - Public datasets mostly provide 2D slices
   - 3D volumetric datasets are scarce and expensive
   - 2D approach leverages larger available datasets

3. CLINICAL WORKFLOW:
   - Radiologists often examine 2D slices individually
   - 2D CNN mimics this clinical practice
   - Easier to interpret and explain

4. PRACTICAL DEPLOYMENT:
   - Real-time inference possible
   - Lower infrastructure costs
   - Suitable for web/mobile deployment

Architecture Overview:
- 4 Convolutional Blocks with increasing filters
- Batch Normalization for training stability
- MaxPooling for spatial downsampling
- Dropout for regularization
- Dense layers for classification

Author: Final Year Project
Date: 2024
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Activation, Input, GlobalAveragePooling2D,
    Add, Multiply, Reshape
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
import numpy as np


class AlzheimerCNN:
    """
    Custom 2D CNN for Alzheimer's Detection
    
    Architecture designed with medical imaging best practices:
    - Gradual feature extraction (32 -> 512 filters)
    - Batch normalization for stable training
    - Strategic dropout for overfitting prevention
    - L2 regularization for weight decay
    """
    
    def __init__(self, 
                 input_shape=(224, 224, 3),
                 num_classes=4,
                 dropout_rate=0.5,
                 l2_lambda=0.001):
        """
        Initialize CNN architecture parameters.
        
        Args:
            input_shape: Input image dimensions (H, W, C)
            num_classes: Number of output classes (4 for Alzheimer's stages)
            dropout_rate: Dropout probability for regularization
            l2_lambda: L2 regularization strength
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.model = None
        
    def build_model(self):
        """
        Build the complete CNN architecture.
        
        ARCHITECTURE EXPLANATION (Layer by Layer):
        ============================================
        
        BLOCK 1: Initial Feature Extraction
        ------------------------------------
        Conv2D(32, 3x3): Detects low-level features (edges, gradients)
        BatchNorm: Normalizes activations, speeds up training
        ReLU: Introduces non-linearity
        MaxPool(2x2): Reduces spatial dimensions by half
        
        BLOCK 2: Low-Level Pattern Detection
        -------------------------------------
        Conv2D(64, 3x3): Detects textures and simple patterns
        BatchNorm + ReLU: Standard activation pipeline
        MaxPool: Further spatial reduction
        
        BLOCK 3: Mid-Level Feature Learning
        ------------------------------------
        Conv2D(128, 3x3): Learns complex patterns (shapes, structures)
        BatchNorm + ReLU
        MaxPool: Continues downsampling
        
        BLOCK 4: High-Level Abstraction
        -------------------------------
        Conv2D(256, 3x3): Abstract brain region features
        Conv2D(512, 3x3): Deep hierarchical features
        BatchNorm + ReLU
        GlobalAveragePooling: Reduces to feature vector
        
        CLASSIFICATION HEAD
        -------------------
        Dense(512): First fully connected layer
        Dropout(0.5): Prevents overfitting (50% neurons dropped)
        Dense(256): Second FC layer
        Dropout(0.3): Lighter dropout
        Dense(4, softmax): Output probabilities for 4 classes
        
        Returns:
            Compiled Keras Model
        """
        
        model = Sequential([
            # ==================== BLOCK 1 ====================
            # Purpose: Initial edge and gradient detection
            Conv2D(32, (3, 3), padding='same', 
                   input_shape=self.input_shape,
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv1_1'),
            BatchNormalization(name='bn1_1'),
            Activation('relu', name='relu1_1'),
            
            Conv2D(32, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv1_2'),
            BatchNormalization(name='bn1_2'),
            Activation('relu', name='relu1_2'),
            
            MaxPooling2D((2, 2), name='pool1'),
            
            # ==================== BLOCK 2 ====================
            # Purpose: Texture and simple pattern detection
            Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv2_1'),
            BatchNormalization(name='bn2_1'),
            Activation('relu', name='relu2_1'),
            
            Conv2D(64, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv2_2'),
            BatchNormalization(name='bn2_2'),
            Activation('relu', name='relu2_2'),
            
            MaxPooling2D((2, 2), name='pool2'),
            
            # ==================== BLOCK 3 ====================
            # Purpose: Complex pattern and shape detection
            Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv3_1'),
            BatchNormalization(name='bn3_1'),
            Activation('relu', name='relu3_1'),
            
            Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv3_2'),
            BatchNormalization(name='bn3_2'),
            Activation('relu', name='relu3_2'),
            
            Conv2D(128, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv3_3'),
            BatchNormalization(name='bn3_3'),
            Activation('relu', name='relu3_3'),
            
            MaxPooling2D((2, 2), name='pool3'),
            
            # ==================== BLOCK 4 ====================
            # Purpose: High-level abstract feature learning
            Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv4_1'),
            BatchNormalization(name='bn4_1'),
            Activation('relu', name='relu4_1'),
            
            Conv2D(256, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv4_2'),
            BatchNormalization(name='bn4_2'),
            Activation('relu', name='relu4_2'),
            
            Conv2D(512, (3, 3), padding='same',
                   kernel_regularizer=l2(self.l2_lambda),
                   name='conv4_3'),
            BatchNormalization(name='bn4_3'),
            Activation('relu', name='relu4_3'),
            
            MaxPooling2D((2, 2), name='pool4'),
            
            # ==================== CLASSIFICATION HEAD ====================
            # Global Average Pooling reduces parameters compared to Flatten
            GlobalAveragePooling2D(name='global_avg_pool'),
            
            # Dense layer 1 with heavy dropout
            Dense(512, kernel_regularizer=l2(self.l2_lambda), name='dense1'),
            BatchNormalization(name='bn_dense1'),
            Activation('relu', name='relu_dense1'),
            Dropout(self.dropout_rate, name='dropout1'),
            
            # Dense layer 2 with lighter dropout
            Dense(256, kernel_regularizer=l2(self.l2_lambda), name='dense2'),
            BatchNormalization(name='bn_dense2'),
            Activation('relu', name='relu_dense2'),
            Dropout(self.dropout_rate * 0.6, name='dropout2'),
            
            # Output layer with softmax for multi-class classification
            Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Initial learning rate for Adam optimizer
        """
        if self.model is None:
            raise ValueError("Build model first before compiling")
            
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print("[INFO] Model compiled successfully")
        print(f"[INFO] Optimizer: Adam (lr={learning_rate})")
        print(f"[INFO] Loss: Categorical Crossentropy")
        
    def get_model_summary(self):
        """
        Print and return model summary for documentation.
        
        Returns:
            String containing model summary
        """
        if self.model is None:
            raise ValueError("Build model first")
            
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        self.model.summary()
        
        # Count parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) 
                                for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) 
                                    for w in self.model.non_trainable_weights])
        
        print("\n" + "="*70)
        print("PARAMETER STATISTICS")
        print("="*70)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Model size (approx): {total_params * 4 / (1024**2):.2f} MB")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params
        }
    
    def save_architecture_diagram(self, filepath='model_architecture.png'):
        """
        Save visual diagram of model architecture.
        
        Args:
            filepath: Path to save the diagram
        """
        if self.model is None:
            raise ValueError("Build model first")
            
        try:
            plot_model(self.model, to_file=filepath, 
                      show_shapes=True, show_layer_names=True,
                      dpi=150)
            print(f"[INFO] Model architecture diagram saved to {filepath}")
        except Exception as e:
            print(f"[WARNING] Could not save diagram: {e}")
            print("[INFO] Install graphviz for diagram generation")


def create_alzheimer_cnn(input_shape=(224, 224, 3), num_classes=4):
    """
    Factory function to create and return compiled CNN model.
    
    Args:
        input_shape: Input image dimensions
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras Model ready for training
    """
    print("="*70)
    print("BUILDING CUSTOM 2D CNN FOR ALZHEIMER'S DETECTION")
    print("="*70)
    
    cnn = AlzheimerCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=0.5,
        l2_lambda=0.001
    )
    
    # Build architecture
    model = cnn.build_model()
    
    # Compile
    cnn.compile_model(learning_rate=0.001)
    
    # Print summary
    cnn.get_model_summary()
    
    # Save diagram
    cnn.save_architecture_diagram()
    
    print("\n" + "="*70)
    print("MODEL READY FOR TRAINING")
    print("="*70)
    
    return model, cnn


# Example usage
if __name__ == "__main__":
    print("[INFO] CNN Model module loaded successfully")
    print("[INFO] Use create_alzheimer_cnn() to build the model")
    
    # Uncomment to test model creation
    # model, cnn = create_alzheimer_cnn()
