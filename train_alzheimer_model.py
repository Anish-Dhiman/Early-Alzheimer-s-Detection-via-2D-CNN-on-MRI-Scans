"""
Simple Training Script for Alzheimer's Detection Model
======================================================

This is the EASIEST way to train your model. Just run this script!

Usage:
    python train_alzheimer_model.py --data_dir /path/to/your/dataset

Example:
    python train_alzheimer_model.py --data_dir ./dataset
    python train_alzheimer_model.py --data_dir C:/Users/YourName/Downloads/Alzheimer_dataset

What this script does:
    1. Loads your MRI images from the dataset folder
    2. Preprocesses images (resize, normalize, augment)
    3. Creates the CNN model
    4. Trains the model
    5. Saves the trained model to ./models/
    6. Shows training graphs
    7. Evaluates the model

Output:
    - Trained model: ./models/best_model.h5
    - Training graphs: ./output/training_curves.png
    - Evaluation results: ./output/evaluation/

Author: Final Year Project
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt
import json
from datetime import datetime


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_dataset_structure(data_dir):
    """Verify dataset folder structure."""
    expected_classes = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
    
    print(f"\n📁 Checking dataset at: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Folder not found: {data_dir}")
        print("\n💡 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset")
        return False
    
    found_classes = []
    for class_name in expected_classes:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"   ✅ {class_name}: {num_images} images")
            found_classes.append(class_name)
        else:
            print(f"   ⚠️  {class_name}: NOT FOUND")
    
    if len(found_classes) == 0:
        print("\n❌ ERROR: No valid class folders found!")
        print("\n💡 Expected structure:")
        print("   dataset/")
        print("   ├── Non-Demented/")
        print("   ├── Very Mild Demented/")
        print("   ├── Mild Demented/")
        print("   └── Moderate Demented/")
        return False
    
    print(f"\n✅ Found {len(found_classes)} classes")
    return True


def create_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    """Create training and validation data generators."""
    
    print_section("STEP 1: DATA PREPROCESSING")
    
    # Training data with augmentation
    print("\n🔄 Creating training generator (with augmentation)...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation data - only rescaling
    print("🔄 Creating validation generator (no augmentation)...")
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42,
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42,
        shuffle=False
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Classes: {train_generator.class_indices}")
    
    return train_generator, val_generator


def compute_class_weights(train_generator):
    """Compute class weights to handle imbalance."""
    
    print("\n⚖️  Computing class weights...")
    
    # Get all labels
    labels = []
    for i in range(len(train_generator)):
        batch_labels = train_generator[i][1]
        labels.extend(np.argmax(batch_labels, axis=1))
        if len(labels) >= train_generator.samples:
            break
    
    labels = np.array(labels[:train_generator.samples])
    
    # Compute weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Print distribution
    class_names = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
    class_counts = Counter(labels)
    
    print("\n   Class Distribution:")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)
        weight = class_weight_dict[i]
        print(f"   • {name}: {count} samples (weight: {weight:.3f})")
    
    return class_weight_dict


def build_model(input_shape=(224, 224, 3), num_classes=4):
    """Build the CNN model."""
    
    print_section("STEP 2: BUILDING MODEL")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
    from tensorflow.keras.regularizers import l2
    
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Block 4
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        # Classification Head
        GlobalAveragePooling2D(),
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    # Print summary
    print("\n📐 Model Architecture:")
    total_params = model.count_params()
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    
    return model


def train_model(model, train_generator, val_generator, class_weights, epochs=50, output_dir='./output'):
    """Train the model."""
    
    print_section("STEP 3: TRAINING MODEL")
    
    # Create output directories
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(models_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {train_generator.batch_size}")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"\n   Press Ctrl+C to stop early\n")
    
    # Train
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(models_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")
    
    return history


def plot_training_curves(history, output_dir='./output'):
    """Plot and save training curves."""
    
    print_section("STEP 4: TRAINING RESULTS")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], 'b-', label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], 'r-', label='Validation Precision')
    axes[1, 0].set_title('Precision', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], 'b-', label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], 'r-', label='Validation Recall')
    axes[1, 1].set_title('Recall', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curves saved to: {plot_path}")
    
    # Show
    plt.show()


def evaluate_model(model, val_generator, output_dir='./output'):
    """Evaluate the trained model."""
    
    print_section("STEP 5: MODEL EVALUATION")
    
    # Generate predictions
    print("\n🔍 Generating predictions...")
    y_true = val_generator.classes
    y_pred_proba = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print("\n📈 Performance Metrics:")
    print(f"   ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ✅ Precision: {precision:.4f}")
    print(f"   ✅ Recall:    {recall:.4f}")
    print(f"   ✅ F1-Score:  {f1:.4f}")
    
    # Classification report
    class_names = ['Non-Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    """Main function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Alzheimer\'s Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to dataset folder (e.g., ./dataset or C:/Users/Name/Downloads/Alzheimer_dataset)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output folder for results (default: ./output)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Print welcome
    print("\n" + "="*70)
    print("  🧠 ALZHEIMER'S DETECTION - MODEL TRAINING")
    print("="*70)
    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    
    # Check dataset
    if not check_dataset_structure(args.data_dir):
        print("\n❌ Dataset check failed. Please fix the issues above.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Data generators
        train_gen, val_gen = create_data_generators(args.data_dir, batch_size=args.batch_size)
        
        # Step 2: Class weights
        class_weights = compute_class_weights(train_gen)
        
        # Step 3: Build model
        model = build_model()
        
        # Step 4: Train
        history = train_model(model, train_gen, val_gen, class_weights, 
                             epochs=args.epochs, output_dir=args.output_dir)
        
        # Step 5: Plot curves
        plot_training_curves(history, output_dir=args.output_dir)
        
        # Step 6: Evaluate
        metrics = evaluate_model(model, val_gen, output_dir=args.output_dir)
        
        # Success
        print_section("✅ TRAINING COMPLETED SUCCESSFULLY")
        print(f"\n📁 Your trained model is saved at:")
        print(f"   {os.path.join(args.output_dir, 'models', 'best_model.h5')}")
        print(f"\n📁 All outputs are in:")
        print(f"   {args.output_dir}")
        print(f"\n🎯 Final Accuracy: {metrics['accuracy']*100:.2f}%")
        print("\n" + "="*70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
