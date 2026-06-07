const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const { v4: uuidv4 } = require('uuid');
const cors = require('cors');

const ImageProcessor = require('./server/modules/imageProcessor');
const Predictor = require('./server/modules/predictor');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Create required directories
async function createDirectories() {
  const dirs = [
    'public/uploads',
    'public/results',
    'models',
  ];
  for (const dir of dirs) {
    try {
      await fs.mkdir(dir, { recursive: true });
      console.log(`[v0] Directory created/verified: ${dir}`);
    } catch (error) {
      console.error(`[v0] Failed to create directory ${dir}:`, error);
    }
  }
}

// Setup multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'public/uploads');
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const name = `upload_${uuidv4()}${ext}`;
    cb(null, name);
  },
});

const fileFilter = (req, file, cb) => {
  const allowedMimes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff'];
  if (allowedMimes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error('Only image files (PNG, JPG, TIFF) are allowed'), false);
  }
};

const upload = multer({ 
  storage, 
  fileFilter,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Initialize modules
const imageProcessor = new ImageProcessor();
const predictor = new Predictor();

let serverReady = false;

// Initialize server
async function initializeServer() {
  try {
    await createDirectories();
    
    // Try to load model if it exists
    const modelPath = path.join('models', 'alzheimers_model');
    try {
      const modelExists = await fs.stat(path.join(modelPath, 'model.json'));
      if (modelExists) {
        await predictor.loadModel(modelPath);
      }
    } catch (error) {
      console.log('[v0] No pre-trained model found, will run in demo mode');
    }

    serverReady = true;
    console.log('[v0] Server initialization complete');
  } catch (error) {
    console.error('[v0] Server initialization failed:', error);
  }
}

// Routes

// Serve index.html for root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/index.html'));
});

// Serve about page if it exists
app.get('/about', (req, res) => {
  res.sendFile(path.join(__dirname, 'public/about.html'), (err) => {
    if (err) {
      res.status(404).send('About page not found');
    }
  });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    server_ready: serverReady,
    model_loaded: predictor.modelLoaded,
    timestamp: new Date().toISOString(),
  });
});

// Prediction endpoint
app.post('/api/predict', upload.single('file'), async (req, res) => {
  try {
    console.log('[v0] Prediction request received');

    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    console.log(`[v0] Processing file: ${req.file.filename}`);

    // Preprocess image
    const imageData = await imageProcessor.preprocessImage(req.file.path);
    console.log('[v0] Image preprocessing complete');

    // Run prediction
    const prediction = await predictor.predict(imageData);
    console.log('[v0] Prediction result:', prediction);

    // Generate Grad-CAM
    let gradcamPath = null;
    const heatmap = await predictor.generateGradCAM(imageData);
    if (heatmap) {
      const gradcamFilename = `gradcam_${uuidv4()}.png`;
      gradcamPath = `results/${gradcamFilename}`;
      await imageProcessor.createHeatmapOverlay(
        req.file.path,
        heatmap,
        path.join('public', gradcamPath)
      );
      console.log(`[v0] Grad-CAM saved to: ${gradcamPath}`);
    }

    // Prepare response
    const response = {
      prediction_result: {
        predicted_class: prediction.predicted_class,
        class_name: prediction.class_info.name,
        confidence: prediction.confidence,
        probabilities: {
          non_demented: prediction.probabilities[0],
          very_mild: prediction.probabilities[1],
          mild: prediction.probabilities[2],
          moderate: prediction.probabilities[3],
        },
        class_info: prediction.class_info,
      },
      images: {
        uploaded: `uploads/${req.file.filename}`,
        gradcam: gradcamPath,
      },
      timestamp: new Date().toISOString(),
    };

    if (prediction.demo_mode) {
      response.demo_mode = true;
      response.message = 'Running in demo mode (no model loaded). Predictions are simulated.';
    }

    res.json(response);
  } catch (error) {
    console.error('[v0] Prediction error:', error);
    res.status(500).json({
      error: 'Prediction failed',
      message: error.message,
    });
  }
});

// Model info endpoint
app.get('/api/model-info', (req, res) => {
  res.json({
    model_loaded: predictor.modelLoaded,
    input_shape: [1, 224, 224, 3],
    classes: predictor.classes,
    running_mode: predictor.modelLoaded ? 'production' : 'demo',
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('[v0] Error:', err);
  if (err instanceof multer.MulterError) {
    if (err.code === 'FILE_TOO_LARGE') {
      return res.status(400).json({ error: 'File too large (max 50MB)' });
    }
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined,
  });
});

// Start server
async function startServer() {
  await initializeServer();

  app.listen(PORT, () => {
    console.log(`
╔════════════════════════════════════════════════════════════╗
║  Early Alzheimer's Detection - JavaScript/Node.js Version  ║
╚════════════════════════════════════════════════════════════╝
    
Server running on: http://localhost:${PORT}
API Endpoints:
  - GET  /api/health          - Server health status
  - POST /api/predict         - Upload image for prediction
  - GET  /api/model-info      - Model information
  
Current Mode: ${predictor.modelLoaded ? 'PRODUCTION (Model Loaded)' : 'DEMO (Simulated Predictions)'}

Open http://localhost:${PORT} in your browser to get started.
    `);
  });
}

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('[v0] SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('[v0] SIGINT received, shutting down gracefully');
  process.exit(0);
});

startServer().catch((error) => {
  console.error('[v0] Failed to start server:', error);
  process.exit(1);
});
