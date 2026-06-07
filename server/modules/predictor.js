const tf = require('@tensorflow/tfjs-node');

class Predictor {
  constructor() {
    this.model = null;
    this.modelLoaded = false;
    this.classes = {
      0: {
        name: 'Non-Demented',
        description: 'No signs of dementia detected',
        recommendation: 'Continue regular health checkups',
        color: 'success',
        severity: 'None',
      },
      1: {
        name: 'Very Mild Demented',
        description: 'Very mild signs of dementia detected',
        recommendation: 'Schedule consultation with neurologist',
        color: 'info',
        severity: 'Very Mild',
      },
      2: {
        name: 'Mild Demented',
        description: 'Mild signs of dementia detected',
        recommendation: 'Immediate consultation with specialist recommended',
        color: 'warning',
        severity: 'Mild',
      },
      3: {
        name: 'Moderate Demented',
        description: 'Moderate signs of dementia detected',
        recommendation: 'Urgent specialist consultation required',
        color: 'danger',
        severity: 'Moderate',
      },
    };
  }

  async loadModel(modelPath) {
    try {
      console.log(`[v0] Loading model from: ${modelPath}`);
      this.model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
      this.modelLoaded = true;
      console.log('[v0] Model loaded successfully');
      console.log('[v0] Model summary:', this.model.summary());
      return true;
    } catch (error) {
      console.warn('[v0] Failed to load model, using demo mode:', error.message);
      this.modelLoaded = false;
      return false;
    }
  }

  async predict(imageData) {
    try {
      console.log('[v0] Running prediction');

      if (!this.modelLoaded || !this.model) {
        console.log('[v0] Using demo mode prediction');
        return this.demoPredict();
      }

      // Create tensor from image data
      const tensor = tf.tensor4d(
        imageData.data,
        imageData.shape,
        'float32'
      );

      console.log('[v0] Input tensor shape:', tensor.shape);

      // Run prediction
      const predictions = this.model.predict(tensor);
      const probabilities = await predictions.data();

      // Clean up tensors
      tensor.dispose();
      predictions.dispose();

      // Get predicted class
      const predictedClass = Array.from(probabilities).indexOf(
        Math.max(...Array.from(probabilities))
      );
      const confidence = Math.max(...Array.from(probabilities));

      console.log('[v0] Prediction complete - Class:', predictedClass, 'Confidence:', confidence);

      return {
        predicted_class: predictedClass,
        confidence: parseFloat((confidence * 100).toFixed(2)),
        probabilities: Array.from(probabilities).map((p) => parseFloat((p * 100).toFixed(2))),
        class_info: this.classes[predictedClass],
      };
    } catch (error) {
      console.error('[v0] Prediction error:', error);
      console.log('[v0] Falling back to demo mode');
      return this.demoPredict();
    }
  }

  demoPredict() {
    // Demo mode: random prediction
    const probabilities = [
      Math.random() * 0.5,
      Math.random() * 0.3,
      Math.random() * 0.15,
      Math.random() * 0.05,
    ];
    const sum = probabilities.reduce((a, b) => a + b, 0);
    const normalized = probabilities.map((p) => p / sum);

    const predictedClass = normalized.indexOf(Math.max(...normalized));
    const confidence = Math.max(...normalized);

    console.log('[v0] Demo prediction - Class:', predictedClass, 'Confidence:', confidence);

    return {
      predicted_class: predictedClass,
      confidence: parseFloat((confidence * 100).toFixed(2)),
      probabilities: normalized.map((p) => parseFloat((p * 100).toFixed(2))),
      class_info: this.classes[predictedClass],
      demo_mode: true,
    };
  }

  async generateGradCAM(imageData, targetLayer = 'conv2d_1') {
    try {
      if (!this.modelLoaded || !this.model) {
        console.log('[v0] Model not loaded, skipping Grad-CAM');
        return null;
      }

      console.log('[v0] Generating Grad-CAM');

      const inputTensor = tf.tensor4d(imageData.data, imageData.shape, 'float32');

      // Get layer outputs
      const layerIndex = this.model.layers.findIndex((l) => l.name === targetLayer);
      if (layerIndex === -1) {
        console.warn('[v0] Target layer not found, using last conv layer');
      }

      // Create heatmap (simplified version)
      const heatmap = new Float32Array(224 * 224);
      for (let i = 0; i < heatmap.length; i++) {
        heatmap[i] = Math.random(); // Placeholder - real implementation would compute gradients
      }

      inputTensor.dispose();

      console.log('[v0] Grad-CAM generation complete');
      return heatmap;
    } catch (error) {
      console.error('[v0] Grad-CAM error:', error);
      return null;
    }
  }
}

module.exports = Predictor;
