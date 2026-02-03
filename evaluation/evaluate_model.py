"""
Model Evaluation for Alzheimer's Detection
==========================================

Comprehensive evaluation module implementing medical AI best practices:
- Standard metrics (Accuracy, Precision, Recall, F1)
- Confusion Matrix visualization
- ROC-AUC for multi-class classification
- Per-class performance analysis
- Medical interpretation of results

WHY EACH METTER MATTERS IN HEALTHCARE:
=======================================

1. ACCURACY:
   - Overall correctness of predictions
   - Can be misleading with imbalanced data
   - Good for balanced datasets

2. PRECISION:
   - Of predicted positives, how many are actual positives
   - High precision = fewer false alarms
   - Important when treatment has side effects

3. RECALL (SENSITIVITY):
   - Of actual positives, how many did we catch
   - HIGH RECALL IS CRITICAL for disease detection
   - Missing a case (false negative) can be life-threatening

4. F1-SCORE:
   - Harmonic mean of precision and recall
   - Balances both metrics
   - Good for imbalanced datasets

5. SPECIFICITY:
   - Of actual negatives, how many did we correctly identify
   - Important for avoiding unnecessary treatments

Author: Final Year Project
Date: 2024
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


class ModelEvaluator:
    """
    Comprehensive model evaluation for medical AI.
    
    Provides both standard ML metrics and medical-specific analysis.
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or [
            'Non-Demented',
            'Very Mild Demented',
            'Mild Demented',
            'Moderate Demented'
        ]
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def predict(self, test_generator):
        """
        Generate predictions on test data.
        
        Args:
            test_generator: Test data generator
        """
        print("[INFO] Generating predictions...")
        
        # Get true labels
        self.y_true = test_generator.classes
        
        # Get predictions (probabilities)
        self.y_pred_proba = self.model.predict(test_generator, verbose=1)
        
        # Get predicted classes
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        
        print(f"[INFO] Predictions generated: {len(self.y_pred)} samples")
        
    def compute_metrics(self):
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Run predict() first")
            
        print("\n" + "="*70)
        print("COMPUTING EVALUATION METRICS")
        print("="*70)
        
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['precision_macro'] = precision_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        self.metrics['recall_macro'] = recall_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        self.metrics['f1_macro'] = f1_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class = precision_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            self.y_true, self.y_pred, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(self.class_names):
            self.metrics[f'precision_{class_name}'] = precision_per_class[i]
            self.metrics[f'recall_{class_name}'] = recall_per_class[i]
            self.metrics[f'f1_{class_name}'] = f1_per_class[i]
            
        # ROC-AUC (multi-class)
        try:
            # Binarize labels for multi-class ROC
            y_true_binarized = label_binarize(
                self.y_true, classes=range(len(self.class_names))
            )
            
            # Compute ROC-AUC for each class
            roc_auc_per_class = {}
            for i, class_name in enumerate(self.class_names):
                fpr, tpr, _ = roc_curve(
                    y_true_binarized[:, i], 
                    self.y_pred_proba[:, i]
                )
                roc_auc_per_class[class_name] = auc(fpr, tpr)
                
            self.metrics['roc_auc_per_class'] = roc_auc_per_class
            self.metrics['roc_auc_macro'] = np.mean(list(roc_auc_per_class.values()))
            
        except Exception as e:
            print(f"[WARNING] Could not compute ROC-AUC: {e}")
            
        # Print results
        self._print_metrics()
        
        return self.metrics
    
    def _print_metrics(self):
        """Print formatted metrics."""
        print("\n" + "="*70)
        print("OVERALL METRICS")
        print("="*70)
        print(f"Accuracy:           {self.metrics['accuracy']:.4f}")
        print(f"Precision (Macro):  {self.metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):     {self.metrics['recall_macro']:.4f}")
        print(f"F1-Score (Macro):   {self.metrics['f1_macro']:.4f}")
        
        if 'roc_auc_macro' in self.metrics:
            print(f"ROC-AUC (Macro):    {self.metrics['roc_auc_macro']:.4f}")
            
        print("\n" + "="*70)
        print("PER-CLASS METRICS")
        print("="*70)
        print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*70)
        for class_name in self.class_names:
            p = self.metrics[f'precision_{class_name}']
            r = self.metrics[f'recall_{class_name}']
            f1 = self.metrics[f'f1_{class_name}']
            print(f"{class_name:<25} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
            
        if 'roc_auc_per_class' in self.metrics:
            print("\n" + "="*70)
            print("PER-CLASS ROC-AUC")
            print("="*70)
            for class_name, auc_score in self.metrics['roc_auc_per_class'].items():
                print(f"{class_name}: {auc_score:.4f}")
                
    def plot_confusion_matrix(self, save_path='confusion_matrix.png', normalize=True):
        """
        Plot and save confusion matrix.
        
        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize values
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Run predict() first")
            
        # Compute confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
            
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Confusion matrix saved to: {save_path}")
        
        # Print medical interpretation
        self._interpret_confusion_matrix(cm, normalize)
        
    def _interpret_confusion_matrix(self, cm, normalized):
        """
        Provide medical interpretation of confusion matrix.
        
        CRITICAL ANALYSIS FOR HEALTHCARE:
        =================================
        
        FALSE NEGATIVES (Missed Cases):
        - Patient has disease but model says healthy
        - MOST DANGEROUS in medical AI
        - Patient doesn't get treatment
        - Disease progresses unchecked
        
        FALSE POSITIVES (False Alarms):
        - Patient is healthy but model says diseased
        - Causes anxiety and unnecessary tests
        - Better than false negatives
        - Can be corrected with follow-up tests
        """
        print("\n" + "="*70)
        print("MEDICAL INTERPRETATION - CONFUSION MATRIX")
        print("="*70)
        
        if normalized:
            cm = (cm * cm.sum(axis=1)[:, np.newaxis]).astype(int)
            
        for i, class_name in enumerate(self.class_names):
            true_positives = cm[i, i]
            false_negatives = cm[i, :].sum() - true_positives
            false_positives = cm[:, i].sum() - true_positives
            
            print(f"\n{class_name}:")
            print(f"  Correctly identified: {true_positives}")
            print(f"  Missed (False Negatives): {false_negatives}")
            print(f"  False Alarms (False Positives): {false_positives}")
            
            # Highlight critical cases
            if i > 0:  # Disease classes
                if false_negatives > 0:
                    print(f"  ⚠️  CRITICAL: {false_negatives} cases of {class_name} were MISSED!")
                    
    def plot_roc_curves(self, save_path='roc_curves.png'):
        """
        Plot ROC curves for all classes.
        
        Args:
            save_path: Path to save the plot
        """
        if self.y_pred_proba is None:
            raise ValueError("Run predict() first")
            
        # Binarize labels
        y_true_binarized = label_binarize(
            self.y_true, classes=range(len(self.class_names))
        )
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'orange', 'red']
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(
                y_true_binarized[:, i], 
                self.y_pred_proba[:, i]
            )
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr, 
                color=color, 
                lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})'
            )
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class Classification', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] ROC curves saved to: {save_path}")
        
    def generate_classification_report(self):
        """
        Generate detailed classification report.
        
        Returns:
            Classification report string
        """
        report = classification_report(
            self.y_true, 
            self.y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        print(report)
        
        return report
    
    def save_metrics(self, save_path='metrics.json'):
        """
        Save all metrics to JSON file.
        
        Args:
            save_path: Path to save metrics
        """
        # Convert numpy types to Python types
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                metrics_serializable[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in value.items()
                }
            elif isinstance(value, (np.floating, np.integer)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
                
        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
            
        print(f"[INFO] Metrics saved to: {save_path}")


def evaluate_alzheimer_model(model, test_generator, output_dir='evaluation_results'):
    """
    Complete evaluation pipeline.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("="*70)
    print("ALZHEIMER'S DETECTION - MODEL EVALUATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model)
    
    # Generate predictions
    evaluator.predict(test_generator)
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    
    # Generate classification report
    evaluator.generate_classification_report()
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        save_path=os.path.join(output_dir, 'confusion_matrix.png'),
        normalize=True
    )
    
    # Plot ROC curves
    evaluator.plot_roc_curves(
        save_path=os.path.join(output_dir, 'roc_curves.png')
    )
    
    # Save metrics
    evaluator.save_metrics(
        save_path=os.path.join(output_dir, 'metrics.json')
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    
    return metrics


# Example usage
if __name__ == "__main__":
    print("[INFO] Evaluation module loaded successfully")
    print("[INFO] Use evaluate_alzheimer_model() to evaluate your model")
