const sharp = require('sharp');
const fs = require('fs').promises;

class ImageProcessor {
  constructor() {
    this.targetSize = 224;
    this.meanValues = [0.485, 0.456, 0.406];
    this.stdValues = [0.229, 0.224, 0.225];
  }

  async preprocessImage(imagePath) {
    try {
      console.log(`[v0] Preprocessing image: ${imagePath}`);

      // Load and convert image to RGB
      const image = sharp(imagePath);
      const metadata = await image.metadata();
      console.log(`[v0] Image metadata:`, metadata);

      // Resize to target size
      const resizedBuffer = await image
        .resize(this.targetSize, this.targetSize, {
          fit: 'fill',
          withoutEnlargement: false,
        })
        .raw()
        .toBuffer({ resolveWithObject: true });

      const imageData = resizedBuffer.data;
      const imageInfo = resizedBuffer.info;

      console.log(`[v0] Resized to ${imageInfo.width}x${imageInfo.height}, channels: ${imageInfo.channels}`);

      // Normalize pixel values to [0, 1]
      const normalized = new Float32Array(this.targetSize * this.targetSize * 3);
      
      for (let i = 0; i < imageData.length; i += imageInfo.channels) {
        const r = imageData[i] / 255.0;
        const g = imageData[i + 1] / 255.0;
        const b = imageData[i + 2] / 255.0;

        // ImageNet normalization
        const idx = (i / imageInfo.channels) * 3;
        normalized[idx] = (r - this.meanValues[0]) / this.stdValues[0];
        normalized[idx + 1] = (g - this.meanValues[1]) / this.stdValues[1];
        normalized[idx + 2] = (b - this.meanValues[2]) / this.stdValues[2];
      }

      console.log(`[v0] Image preprocessing complete`);
      return {
        data: normalized,
        shape: [1, this.targetSize, this.targetSize, 3],
        originalPath: imagePath,
      };
    } catch (error) {
      console.error('[v0] Image preprocessing error:', error);
      throw new Error(`Failed to preprocess image: ${error.message}`);
    }
  }

  async createHeatmapOverlay(imagePath, heatmapData, outputPath) {
    try {
      console.log(`[v0] Creating heatmap overlay`);

      // Load original image
      const image = await sharp(imagePath).resize(224, 224).toBuffer();

      // Create heatmap visualization
      const heatmapBuffer = Buffer.alloc(224 * 224 * 4);
      for (let i = 0; i < 224 * 224; i++) {
        const intensity = Math.min(255, Math.max(0, heatmapData[i] * 255));
        heatmapBuffer[i * 4] = intensity; // R
        heatmapBuffer[i * 4 + 1] = 0; // G
        heatmapBuffer[i * 4 + 2] = 255 - intensity; // B
        heatmapBuffer[i * 4 + 3] = 255; // A
      }

      // Create heatmap image
      const heatmapImage = sharp(heatmapBuffer, {
        raw: { width: 224, height: 224, channels: 4 },
      });

      // Composite images
      const composite = await sharp(image)
        .composite([{ input: await heatmapImage.toBuffer(), blend: 'overlay' }])
        .png()
        .toFile(outputPath);

      console.log(`[v0] Heatmap overlay created: ${outputPath}`);
      return outputPath;
    } catch (error) {
      console.error('[v0] Heatmap creation error:', error);
      throw new Error(`Failed to create heatmap: ${error.message}`);
    }
  }
}

module.exports = ImageProcessor;
