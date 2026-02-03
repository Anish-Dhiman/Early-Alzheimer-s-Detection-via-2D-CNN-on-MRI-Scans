"""
Grad-CAM Explainability for Alzheimer's Detection
=================================================

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to visualize which regions of MRI scans the CNN focuses on for predictions.

WHY GRAD-CAM IS CRITICAL FOR MEDICAL AI:
=========================================

1. TRUST & TRANSPARENCY:
   - Doctors need to understand WHY the AI made a decision
   - Black-box models are not accepted in healthcare
   - Grad-CAM provides visual explanations

2. CLINICAL VALIDATION:
   - Heatmaps should align with known disease biomarkers
   - Hippocampus atrophy is key indicator of Alzheimer's
   - Grad-CAM can validate if model learns correct features

3. ERROR ANALYSIS:
   - When model is wrong, Grad-CAM shows what it looked at
   - Helps identify failure modes
   - Guides model improvement

4. REGULATORY COMPLIANCE:
   - FDA/CE marking requires explainability
   - Grad-CAM provides documentation for approval

GRAD-CAM ALGORITHM:
===================
1. Forward pass to get class prediction
2. Compute gradient of target class w.r.t. feature maps
3. Global Average Pool gradients to get weights
4. Weighted combination of feature maps
5. ReLU to keep only positive contributions
6. Upsample to original image size
7. Overlay on original image

Reference: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization" ICCV 2017

Author: Final Year Project
Date: 2024
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN explainability.
    
    Generates heatmaps showing which image regions influence predictions.
    """
    
    def __init__(self, model, last_conv_layer_name=None):
        """
        Initialize Grad-CAM with model.
        
        Args:
            model: Trained Keras model
            last_conv_layer_name: Name of last convolutional layer
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        
        # Find last conv layer if not specified
        if self.last_conv_layer_name is None:
            self.last_conv_layer_name = self._find_last_conv_layer()
            
        print(f"[INFO] Grad-CAM using layer: {self.last_conv_layer_name}")
        
        # Create model that outputs both predictions and conv features
        self.grad_model = Model(
            inputs=[model.inputs],
            outputs=[
                model.get_layer(self.last_conv_layer_name).output,
                model.output
            ]
        )
        
    def _find_last_conv_layer(self):
        """Automatically find the last convolutional layer."""
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("No convolutional layer found in model")
        
    def compute_heatmap(self, image, class_index=None):
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Input image (preprocessed, with batch dimension)
            class_index: Target class index (None = predicted class)
            
        Returns:
            Heatmap array (0-255 range)
        """
        # Convert image to tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Record operations for gradient computation
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            
            # Forward pass
            conv_outputs, predictions = self.grad_model(image_tensor)
            
            # Get target class
            if class_index is None:
                class_index = tf.argmax(predictions[0])
                
            # Get score for target class
            target_score = predictions[:, class_index]
            
        # Compute gradients
        gradients = tape.gradient(target_score, conv_outputs)
        
        # Global Average Pooling of gradients
        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        
        # Weighted combination of feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        # Apply ReLU (keep only positive contributions)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to 0-1
        heatmap = heatmap / (np.max(heatmap) + 1e-10)
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        
        # Convert to 0-255 range
        heatmap = np.uint8(255 * heatmap)
        
        return heatmap, class_index.numpy(), predictions[0].numpy()
        
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (0-255 range)
            heatmap: Grad-CAM heatmap
            alpha: Transparency of overlay (0-1)
            colormap: OpenCV colormap
            
        Returns:
            Superimposed image
        """
        # Ensure image is in correct format
        if image.max() <= 1.0:
            image = np.uint8(image * 255)
            
        # Remove batch dimension if present
        if len(image.shape) == 4:
            image = image[0]
            
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return superimposed
        
    def visualize(self, 
                  image_path, 
                  preprocessor,
                  class_names=None,
                  save_path=None,
                  alpha=0.4):
        """
        Complete visualization pipeline for a single image.
        
        Args:
            image_path: Path to image file
            preprocessor: MRIPreprocessor instance
            class_names: List of class names
            save_path: Path to save visualization
            alpha: Overlay transparency
            
        Returns:
            Dictionary with results
        """
        if class_names is None:
            class_names = [
                'Non-Demented',
                'Very Mild Demented',
                'Mild Demented',
                'Moderate Demented'
            ]
            
        # Load and preprocess image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (224, 224))
        
        preprocessed_img = preprocessor.preprocess_single_image(image_path)
        
        # Compute Grad-CAM
        heatmap, predicted_class, probabilities = self.compute_heatmap(preprocessed_img)
        
        # Create overlay
        overlay = self.overlay_heatmap(original_img, heatmap, alpha=alpha)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original MRI Scan', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay: {class_names[predicted_class]}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Confidence bar chart
        colors = ['green' if i == predicted_class else 'gray' 
                  for i in range(len(class_names))]
        bars = axes[3].barh(class_names, probabilities * 100, color=colors)
        axes[3].set_xlabel('Confidence (%)', fontsize=11)
        axes[3].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
        axes[3].set_xlim(0, 100)
        
        # Add percentage labels
        for bar, prob in zip(bars, probabilities):
            axes[3].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{prob*100:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Grad-CAM visualization saved to: {save_path}")
            
        plt.close()
        
        return {
            'predicted_class': class_names[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(class_names, probabilities)
            },
            'heatmap': heatmap,
            'overlay': overlay
        }
        
    def analyze_multiple(self, 
                        image_paths, 
                        preprocessor,
                        class_names=None,
                        output_dir='gradcam_results'):
        """
        Analyze multiple images and save results.
        
        Args:
            image_paths: List of image paths
            preprocessor: MRIPreprocessor instance
            class_names: List of class names
            output_dir: Directory to save results
            
        Returns:
            List of result dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"[INFO] Processing {i+1}/{len(image_paths)}: {image_path}")
            
            save_path = os.path.join(output_dir, f'gradcam_{i+1}.png')
            
            result = self.visualize(
                image_path=image_path,
                preprocessor=preprocessor,
                class_names=class_names,
                save_path=save_path
            )
            
            result['image_path'] = image_path
            results.append(result)
            
        print(f"[INFO] All Grad-CAM visualizations saved to: {output_dir}")
        
        return results
        
    def interpret_heatmap(self, heatmap, class_name):
        """
        Provide medical interpretation of Grad-CAM heatmap.
        
        Args:
            heatmap: Grad-CAM heatmap
            class_name: Predicted class name
            
        Returns:
            Interpretation string
        """
        # Find regions of high activation
        threshold = np.percentile(heatmap, 90)
        high_activation = heatmap > threshold
        
        # Calculate activation statistics
        total_pixels = heatmap.size
        activated_pixels = np.sum(high_activation)
        activation_percentage = (activated_pixels / total_pixels) * 100
        
        interpretation = f"""
GRAD-CAM INTERPRETATION FOR: {class_name}
{'='*60}

Heatmap Analysis:
- High activation covers {activation_percentage:.1f}% of the image
- Peak activation intensity: {heatmap.max()}/255
- Mean activation: {heatmap.mean():.1f}/255

Medical Interpretation:
"""
        
        if 'Non-Demented' in class_name:
            interpretation += """
- Model focuses on overall brain structure
- No specific atrophy regions highlighted
- Consistent with healthy brain appearance
"""
        elif 'Very Mild' in class_name:
            interpretation += """
- Model may detect subtle changes in medial temporal lobe
- Early hippocampal atrophy might be visible
- Requires careful clinical correlation
"""
        elif 'Mild' in class_name:
            interpretation += """
- Model highlights hippocampal and parahippocampal regions
- Moderate atrophy patterns visible
- Enlarged ventricles may be detected
"""
        elif 'Moderate' in class_name:
            interpretation += """
- Strong activation in temporal and parietal lobes
- Significant hippocampal atrophy detected
- Widespread cortical changes visible
"""
            
        interpretation += """
Clinical Correlation:
- Grad-CAM regions should align with known Alzheimer's biomarkers
- Hippocampal atrophy is the primary indicator
- Ventricular enlargement supports atrophy assessment
- Always correlate with clinical symptoms and other tests
"""
        
        return interpretation


def create_gradcam_explainer(model, last_conv_layer_name=None):
    """
    Factory function to create Grad-CAM explainer.
    
    Args:
        model: Trained Keras model
        last_conv_layer_name: Name of last conv layer (auto-detected if None)
        
    Returns:
        GradCAM instance
    """
    print("="*70)
    print("INITIALIZING GRAD-CAM EXPLAINABILITY")
    print("="*70)
    
    gradcam = GradCAM(model, last_conv_layer_name)
    
    print("[INFO] Grad-CAM explainer ready")
    print("[INFO] Use visualize() or analyze_multiple() to generate explanations")
    
    return gradcam


# Example usage
if __name__ == "__main__":
    print("[INFO] Grad-CAM module loaded successfully")
    print("[INFO] Use create_gradcam_explainer() to create explainer")
