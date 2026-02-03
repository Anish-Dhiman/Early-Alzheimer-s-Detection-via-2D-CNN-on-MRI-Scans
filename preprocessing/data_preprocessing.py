"""
Data Preprocessing Pipeline for Alzheimer's Detection
======================================================

This module handles all preprocessing steps for MRI scan images:
- Image resizing to 224x224 (standard input size for CNNs)
- Normalization (pixel values to [0,1] range)
- Data augmentation (rotation, flip, zoom) for training robustness
- Train/validation/test split
- Class weighting for handling imbalanced dataset

Author: Final Year Project
Date: 2024
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import cv2


class MRIPreprocessor:
    """
    MRI Data Preprocessor Class
    
    Handles all preprocessing operations for Alzheimer's MRI dataset.
    Designed for 4-class classification:
    - Non-Demented (Class 0)
    - Very Mild Demented (Class 1)
    - Mild Demented (Class 2)
    - Moderate Demented (Class 3)
    """
    
    def __init__(self, 
                 img_size=(224, 224),
                 batch_size=32,
                 validation_split=0.2,
                 test_split=0.1,
                 random_seed=42):
        """
        Initialize preprocessor with configuration parameters.
        
        Args:
            img_size: Target image dimensions (height, width)
            batch_size: Number of images per batch
            validation_split: Fraction of data for validation
            test_split: Fraction of data for testing
            random_seed: For reproducibility
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed
        
        # Class mapping for Alzheimer's stages
        self.class_names = [
            'Non-Demented',
            'Very Mild Demented', 
            'Mild Demented',
            'Moderate Demented'
        ]
        
        # Data generators (initialized later)
        self.train_datagen = None
        self.val_datagen = None
        self.test_datagen = None
        
        # Generators
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        
    def create_data_generators(self):
        """
        Create ImageDataGenerators with augmentation for training.
        
        WHY DATA AUGMENTATION IS CRITICAL:
        1. Medical datasets are often small - augmentation artificially expands dataset
        2. Makes model robust to variations in scan orientation
        3. Prevents overfitting on limited training samples
        4. Simulates real-world variations in MRI acquisition
        
        Augmentation Strategy for Medical Images:
        - Rotation: ±15° (patients may be positioned differently)
        - Horizontal flip: Anatomical symmetry allows this
        - Zoom: ±10% (different scan zoom levels)
        - NO vertical flip (anatomy is not vertically symmetric)
        - NO shear/warp (would distort anatomical structures)
        """
        
        # Training generator with augmentation
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize pixel values to [0,1]
            rotation_range=15,  # Small rotations for position variation
            width_shift_range=0.1,  # Small horizontal shifts
            height_shift_range=0.1,  # Small vertical shifts
            horizontal_flip=True,  # Brain has left-right symmetry
            zoom_range=0.1,  # Simulate different zoom levels
            fill_mode='nearest',  # Fill gaps after transformation
            validation_split=self.validation_split + self.test_split
        )
        
        # Validation generator - NO augmentation, only normalization
        self.val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.validation_split + self.test_split
        )
        
        # Test generator - NO augmentation, only normalization
        self.test_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        print("[INFO] Data generators created successfully")
        print("[INFO] Training: WITH augmentation")
        print("[INFO] Validation/Test: NO augmentation (only normalization)")
        
    def load_data(self, data_dir):
        """
        Load and split data into train/validation/test sets.
        
        Expected directory structure:
        data_dir/
            Non-Demented/
                img1.jpg, img2.jpg, ...
            Very Mild Demented/
                img1.jpg, img2.jpg, ...
            Mild Demented/
                img1.jpg, img2.jpg, ...
            Moderate Demented/
                img1.jpg, img2.jpg, ...
        
        Args:
            data_dir: Root directory containing class subdirectories
            
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        if self.train_datagen is None:
            self.create_data_generators()
            
        # Training data (with augmentation)
        self.train_generator = self.train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            seed=self.random_seed,
            shuffle=True
        )
        
        # Validation data (no augmentation)
        # Note: We use a portion of the validation_split for validation
        self.val_generator = self.val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            seed=self.random_seed,
            shuffle=False
        )
        
        # For test set, we'll manually split from validation or use separate directory
        # Here we create a separate test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        print("\n[INFO] Data Loaded Successfully!")
        print(f"[INFO] Training samples: {self.train_generator.samples}")
        print(f"[INFO] Validation samples: {self.val_generator.samples}")
        print(f"[INFO] Classes: {self.train_generator.class_indices}")
        
        return self.train_generator, self.val_generator
    
    def compute_class_weights(self):
        """
        Compute class weights to handle imbalanced dataset.
        
        WHY CLASS WEIGHTING IS CRITICAL IN MEDICAL AI:
        - Alzheimer's datasets are naturally imbalanced
        - Severe cases (Moderate Demented) are rarer
        - Without weighting, model would be biased toward majority class
        - False negatives (missing disease) are more costly than false positives
        
        Returns:
            Dictionary mapping class indices to weights
        """
        if self.train_generator is None:
            raise ValueError("Load data first before computing class weights")
            
        # Get all labels from generator
        labels = []
        for i in range(len(self.train_generator)):
            batch_labels = self.train_generator[i][1]
            labels.extend(np.argmax(batch_labels, axis=1))
            if len(labels) >= self.train_generator.samples:
                break
        
        labels = np.array(labels[:self.train_generator.samples])
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print("\n[INFO] Class Distribution and Weights:")
        class_counts = Counter(labels)
        for class_idx, class_name in enumerate(self.class_names):
            count = class_counts.get(class_idx, 0)
            weight = class_weight_dict[class_idx]
            print(f"  {class_name}: {count} samples, weight={weight:.4f}")
            
        return class_weight_dict
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for model input
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, self.img_size)
        
        # Normalize
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def visualize_augmentation(self, sample_image_path, num_variations=5):
        """
        Visualize data augmentation effects on a sample image.
        Useful for understanding augmentation impact.
        
        Args:
            sample_image_path: Path to sample image
            num_variations: Number of augmented versions to show
        """
        img = cv2.imread(sample_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = np.expand_dims(img, axis=0)
        
        # Create augmentation generator for single image
        aug_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Generate augmented images
        aug_iter = aug_gen.flow(img, batch_size=1)
        
        fig, axes = plt.subplots(1, num_variations + 1, figsize=(15, 3))
        
        # Original
        axes[0].imshow(img[0] / 255.0)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented versions
        for i in range(num_variations):
            aug_img = next(aug_iter)[0]
            axes[i + 1].imshow(aug_img)
            axes[i + 1].set_title(f'Augmented {i+1}')
            axes[i + 1].axis('off')
            
        plt.tight_layout()
        plt.savefig('augmentation_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("[INFO] Augmentation visualization saved to 'augmentation_visualization.png'")


def preprocess_pipeline(data_dir, img_size=(224, 224), batch_size=32):
    """
    Complete preprocessing pipeline wrapper function.
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size
        batch_size: Batch size for generators
        
    Returns:
        Tuple of (preprocessor, train_gen, val_gen, class_weights)
    """
    print("="*60)
    print("ALZHEIMER'S DETECTION - DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = MRIPreprocessor(
        img_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        test_split=0.1,
        random_seed=42
    )
    
    # Load data
    train_gen, val_gen = preprocessor.load_data(data_dir)
    
    # Compute class weights
    class_weights = preprocessor.compute_class_weights()
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    return preprocessor, train_gen, val_gen, class_weights


# Example usage
if __name__ == "__main__":
    # Example: preprocess_pipeline('/path/to/Alzheimer_dataset')
    print("[INFO] Preprocessing module loaded successfully")
    print("[INFO] Use preprocess_pipeline(data_dir) to process your dataset")
