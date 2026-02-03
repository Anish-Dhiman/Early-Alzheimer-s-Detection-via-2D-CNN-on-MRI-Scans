"""
Flask Web Application for Alzheimer's Detection
===============================================

A doctor-friendly web interface for:
- Uploading MRI scan images
- Getting AI-powered predictions
- Viewing Grad-CAM explanations
- Understanding confidence scores

Features:
- Clean, intuitive UI
- Real-time predictions
- Visual explanations
- Medical interpretation

Author: Final Year Project
Date: 2024
"""

import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import uuid
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_preprocessing import MRIPreprocessor
from explainability.gradcam import GradCAM

# Flask App Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'alzheimers-detection-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Class names
CLASS_NAMES = [
    'Non-Demented',
    'Very Mild Demented',
    'Mild Demented',
    'Moderate Demented'
]

# Class descriptions for UI
CLASS_DESCRIPTIONS = {
    'Non-Demented': {
        'description': 'No signs of dementia. Brain structure appears normal.',
        'recommendation': 'Regular check-ups recommended. No immediate concern.',
        'color': '#28a745',
        'severity': 'Normal'
    },
    'Very Mild Demented': {
        'description': 'Early signs of cognitive decline. Subtle brain changes.',
        'recommendation': 'Monitor closely. Consider neuropsychological testing.',
        'color': '#ffc107',
        'severity': 'Mild Concern'
    },
    'Mild Demented': {
        'description': 'Noticeable memory and cognitive impairment.',
        'recommendation': 'Clinical evaluation advised. Consider treatment options.',
        'color': '#fd7e14',
        'severity': 'Moderate Concern'
    },
    'Moderate Demented': {
        'description': 'Significant cognitive decline. Clear brain atrophy.',
        'recommendation': 'Immediate clinical intervention required.',
        'color': '#dc3545',
        'severity': 'High Concern'
    }
}


class AlzheimerApp:
    """
    Main application class handling model loading and predictions.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize application with model.
        
        Args:
            model_path: Path to trained model file
        """
        self.model = None
        self.preprocessor = None
        self.gradcam = None
        self.model_loaded = False
        
        # Initialize preprocessor
        self.preprocessor = MRIPreprocessor(img_size=(224, 224))
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("[WARNING] No model loaded. Running in demo mode.")
            
    def load_model(self, model_path):
        """
        Load trained model.
        
        Args:
            model_path: Path to .h5 or SavedModel
        """
        try:
            print(f"[INFO] Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
            
            # Initialize Grad-CAM
            self.gradcam = GradCAM(self.model)
            
            print("[INFO] Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model_loaded = False
            return False
            
    def predict(self, image_path):
        """
        Make prediction on an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        if not self.model_loaded:
            # Return demo prediction
            return self._demo_prediction(image_path)
            
        try:
            # Preprocess image
            preprocessed = self.preprocessor.preprocess_single_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(preprocessed, verbose=0)
            probabilities = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            
            # Generate Grad-CAM
            result_id = str(uuid.uuid4())[:8]
            gradcam_path = os.path.join(app.config['RESULTS_FOLDER'], 
                                       f'gradcam_{result_id}.png')
            
            gradcam_result = self.gradcam.visualize(
                image_path=image_path,
                preprocessor=self.preprocessor,
                class_names=CLASS_NAMES,
                save_path=gradcam_path
            )
            
            # Prepare result
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': {
                    name: float(prob) 
                    for name, prob in zip(CLASS_NAMES, probabilities)
                },
                'class_info': CLASS_DESCRIPTIONS[predicted_class],
                'gradcam_image': f'results/gradcam_{result_id}.png',
                'result_id': result_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def _demo_prediction(self, image_path):
        """
        Generate demo prediction for testing without model.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Demo prediction result
        """
        # Random prediction for demo
        probabilities = np.random.dirichlet(np.ones(4)) * 0.8 + 0.05
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        result_id = str(uuid.uuid4())[:8]
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': float(probabilities[predicted_class_idx]),
            'all_probabilities': {
                name: float(prob) 
                for name, prob in zip(CLASS_NAMES, probabilities)
            },
            'class_info': CLASS_DESCRIPTIONS[predicted_class],
            'gradcam_image': None,
            'result_id': result_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'demo_mode': True
        }


# Initialize app
app_core = AlzheimerApp()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page with upload form."""
    return render_template('index.html', 
                          model_loaded=app_core.model_loaded,
                          class_names=CLASS_NAMES)


@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction
        result = app_core.predict(filepath)
        result['uploaded_image'] = f'uploads/{unique_filename}'
        
        return jsonify(result)
    else:
        return jsonify({
            'success': False, 
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """API endpoint to load a model."""
    data = request.get_json()
    model_path = data.get('model_path')
    
    if not model_path:
        return jsonify({'success': False, 'error': 'No model path provided'}), 400
        
    success = app_core.load_model(model_path)
    
    return jsonify({
        'success': success,
        'message': 'Model loaded successfully' if success else 'Failed to load model'
    })


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': app_core.model_loaded,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error_code=404, 
                          error_message='Page not found'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_code=500,
                          error_message='Internal server error'), 500


def initialize_app(model_path=None):
    """
    Initialize application with model.
    
    Args:
        model_path: Path to trained model
    """
    global app_core

    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    DEFAULT_MODEL_PATH = os.path.join("models", "best_model.h5")

    # Priority 1: model path from command line
    if model_path and os.path.exists(model_path):
        app_core.load_model(model_path)

    # Priority 2: auto-load from models folder
    elif os.path.exists(DEFAULT_MODEL_PATH):
        app_core.load_model(DEFAULT_MODEL_PATH)

    print("\n" + "="*70)
    print("ALZHEIMER'S DETECTION WEB APP")
    print("="*70)
    print(f"Model Status: {'Loaded' if app_core.model_loaded else 'Demo Mode'}")
    print(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Results Folder: {app.config['RESULTS_FOLDER']}")
    print("="*70 + "\n")



# Run application
if __name__ == '__main__':
    # Initialize with model (if available)
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    initialize_app(model_path)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
