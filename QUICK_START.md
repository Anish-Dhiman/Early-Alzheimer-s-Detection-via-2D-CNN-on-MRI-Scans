# Quick Start Guide - JavaScript Version

## 🚀 Get Started in 2 Minutes

### Prerequisites
- Node.js 18+ installed
- ~500MB disk space
- ~500MB RAM

### Step 1: Install Dependencies (30 seconds)
```bash
cd /vercel/share/v0-project
npm install
```

### Step 2: Start the Server (10 seconds)
```bash
npm start
```

You'll see:
```
╔════════════════════════════════════════════════════════════════════════════╗
║  Early Alzheimer's Detection - JavaScript/Node.js Version                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Server running on: http://localhost:5000
Current Mode: DEMO (Simulated Predictions)
```

### Step 3: Open in Browser (5 seconds)
Navigate to: **http://localhost:5000**

You should see:
- 🧠 "Alzheimer's Detection" title
- 📤 Upload area with drag-and-drop
- 🔗 Navigation to "About" page

### Step 4: Test Upload
1. Click "Choose File" or drag an image
2. Select any MRI or medical image file
3. See instant prediction results with:
   - Confidence percentage
   - Risk classification
   - Clinical recommendation

Done! 🎉

---

## ❓ Common Questions

### Q: What if the server doesn't start?
**A:** Check the error message. Common fixes:
```bash
# Port already in use?
lsof -ti:5000 | xargs kill -9

# Node version too old?
node --version  # Should be v18.0.0+

# Dependencies missing?
rm -rf node_modules package-lock.json
npm install
```

### Q: Why are predictions random?
**A:** You're in **demo mode**. No trained model is loaded. This is normal! You can:
- Use the app to test functionality
- Upload your own trained model later
- Check the "About" page for more info

### Q: How do I use my trained model?
**A:** Convert it to TensorFlow.js format:

```bash
# Install the converter
pip install tensorflowjs

# Convert your .h5 file
tensorflowjs_converter --input_format=keras your_model.h5 models/alzheimers_model

# Restart the server
npm start
```

Your model will auto-load and enable production mode!

### Q: Can I upload large images?
**A:** Yes! Up to 50MB. The app automatically resizes to 224x224 for processing.

### Q: Where are my uploads stored?
**A:** In `/public/uploads/` with unique filenames. Each prediction creates:
- `upload_[UUID].png` - Your original image
- `gradcam_[UUID].png` - AI attention heatmap (if model loaded)

### Q: How do I see debug logs?
**A:** All logs are prefixed with `[v0]`:

```bash
npm start 2>&1 | grep "\[v0\]"
```

---

## 📊 API Examples

### Health Check
```bash
curl http://localhost:5000/api/health

# Response:
# {"status":"ok","server_ready":true,"model_loaded":false}
```

### Predict with Image
```bash
curl -X POST -F "file=@test_image.png" http://localhost:5000/api/predict

# Response:
# {
#   "prediction_result": {
#     "predicted_class": 0,
#     "class_name": "Non-Demented",
#     "confidence": 61.92,
#     ...
#   },
#   "demo_mode": true
# }
```

### Model Information
```bash
curl http://localhost:5000/api/model-info

# Response:
# {
#   "model_loaded": false,
#   "running_mode": "demo",
#   "classes": {...}
# }
```

---

## 🎯 UI Features

### Home Page (`/`)
- **Upload Section**: Drag-drop or click to upload
- **Results**: Shows prediction with:
  - Class name (Non-Demented, Very Mild, Mild, Moderate)
  - Confidence percentage
  - Severity level badge
  - Clinical recommendation
  - Probability breakdown for all classes
  - Original and Grad-CAM images (side-by-side)
- **Demo Banner**: Appears when no model is loaded

### About Page (`/about`)
- Project overview
- Technology stack
- System pipeline
- CNN architecture details
- Classification classes
- Future enhancements
- Important disclaimer

---

## 📁 Project Structure

```
/vercel/share/v0-project/
├── server.js                    # Main Express app
├── package.json                 # Dependencies
├── README_JAVASCRIPT.md         # Full documentation
├── CONVERSION_SUMMARY.md        # What was converted
├── QUICK_START.md              # This file
│
├── public/                      # Static files
│   ├── index.html              # Home page
│   ├── about.html              # About page
│   ├── uploads/                # Your uploaded images
│   └── results/                # Grad-CAM heatmaps
│
├── server/modules/             # Backend logic
│   ├── imageProcessor.js       # Image handling
│   ├── predictor.js            # AI predictions
│   └── gradcam.js              # Explainability
│
├── models/                      # ML models go here
│   └── (empty until you add model)
│
└── node_modules/               # Dependencies (auto-installed)
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot find module` | Run `npm install` |
| `Port 5000 in use` | Kill process: `lsof -ti:5000 \| xargs kill -9` |
| `Out of memory` | Node uses ~300-500MB. Check available RAM. |
| `Slow predictions` | Normal in JS (500-1000ms). Use GPU-enabled setup for faster speeds. |
| `File not uploading` | Check file size (<50MB) and format (PNG, JPG, TIFF) |
| `Images not loading` | Clear browser cache or use incognito mode |

---

## 🚀 Deployment

### Run in Background (Linux/Mac)
```bash
npm start &
# Server runs in background, save PID
disown
# To stop: kill [PID]
```

### Use PM2 (Recommended)
```bash
npm install -g pm2
pm2 start server.js --name "alzheimers"
pm2 save           # Persist on restart
pm2 startup        # Auto-start on reboot
```

### Docker
```bash
docker build -t alzheimers-app .
docker run -p 5000:5000 alzheimers-app
```

### Vercel (Serverless)
```bash
npm install -g vercel
vercel --prod
```

---

## 📈 Performance Tips

1. **Use PNG images**: Faster than JPG for processing
2. **Compress images**: Reduce file size before upload (still works auto-resizes)
3. **Browser caching**: Results cached automatically
4. **Batch uploads**: API handles concurrent requests
5. **Monitor memory**: Check system resources if slow

---

## 🎓 Learning Resources

- **Express.js**: https://expressjs.com/
- **TensorFlow.js**: https://www.tensorflow.org/js
- **Sharp Documentation**: https://sharp.pixelplumbing.com/
- **Bootstrap 5**: https://getbootstrap.com/

---

## 📞 Need Help?

1. **Check console logs**: Filter by `[v0]` prefix
2. **Read README_JAVASCRIPT.md**: Full detailed docs
3. **Test with curl**: API works independently of UI
4. **Check node_modules**: Verify all packages installed
5. **Review error messages**: Always descriptive

---

## ✅ Verification Checklist

After starting the server, verify:

- [ ] Server output shows "Server running on: http://localhost:5000"
- [ ] Can access http://localhost:5000 in browser
- [ ] Home page loads with upload area
- [ ] About page accessible
- [ ] Can upload an image
- [ ] Prediction results display
- [ ] Demo mode banner visible
- [ ] No JavaScript errors in browser console
- [ ] API endpoints respond (test with curl)

If all checkmarks pass, you're ready to use the app! 🎉

---

## 🔄 Update/Reinstall

To reset to a clean state:
```bash
# Kill server
pkill -f "node server.js"

# Clean installation
rm -rf node_modules package-lock.json
npm install

# Start fresh
npm start
```

---

**Status**: ✅ Ready to Use!

Version: 1.0.0 (JavaScript/Node.js)
Framework: Express 4.18 + TensorFlow.js 4.11
Updated: June 7, 2026
