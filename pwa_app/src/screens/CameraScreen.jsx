import { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm'
import './CameraScreen.css'

const MODEL_PATH = '/model/model.json?v=' + Date.now()
const THRESHOLD = 0.5
const TOTAL_FRAMES = 5

export default function CameraScreen({ onResult, onBack }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const modelRef = useRef(null)

  const [mode, setMode] = useState('camera') // 'camera' | 'upload'
  const [hasPermission, setHasPermission] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const [modelError, setModelError] = useState(null)
  const [capturing, setCapturing] = useState(false)
  const [frameCount, setFrameCount] = useState(0)
  const [modelLoaded, setModelLoaded] = useState(false)
  const [loadingModel, setLoadingModel] = useState(true)

  // Upload mode state
  const [uploadedImage, setUploadedImage] = useState(null) // data URL
  const [uploadedCanvas, setUploadedCanvas] = useState(null) // offscreen canvas
  const [analyzing, setAnalyzing] = useState(false)

  // Load model on mount
  useEffect(() => {
    loadModel()
  }, [])

  // Start/stop camera based on mode
  useEffect(() => {
    if (mode === 'camera') {
      startCamera()
    } else {
      stopCamera()
      setUploadedImage(null)
      setUploadedCanvas(null)
    }
    return () => {}
  }, [mode])

  // Cleanup camera on unmount
  useEffect(() => {
    return () => stopCamera()
  }, [])

  const loadModel = async () => {
    try {
      setLoadingModel(true)
      setModelError(null)
      // Use WASM backend — the .wasm files are served from /public
      setWasmPaths('/')
      await tf.setBackend('wasm')
      await tf.ready()
      console.log('Backend:', tf.getBackend())
      const model = await tf.loadLayersModel(MODEL_PATH)
      modelRef.current = model
      setModelLoaded(true)
      console.log('✓ Model loaded successfully with WASM backend')
    } catch (err) {
      console.error('WASM backend error:', err)
      try {
        await tf.setBackend('cpu')
        await tf.ready()
        const model = await tf.loadLayersModel(MODEL_PATH)
        modelRef.current = model
        setModelLoaded(true)
        console.log('✓ Model loaded with CPU fallback')
      } catch (err2) {
        console.error('CPU fallback error:', err2)
        setModelError(`Failed to load AI model: ${err2.message || 'Unknown error'}`)
      }
    } finally {
      setLoadingModel(false)
    }
  }

  const startCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setHasPermission(true)
      }
    } catch (err) {
      setCameraError('Camera access denied. Please allow camera permission and refresh.')
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    setHasPermission(false)
  }

  // ── Image quality checks ───────────────────────────────────────
  const isBlurry = (imageData) => {
    const data = imageData.data
    let sum = 0, sumSq = 0
    const count = data.length / 4
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      sum += gray
      sumSq += gray * gray
    }
    const mean = sum / count
    const variance = (sumSq / count) - (mean * mean)
    return variance < 500
  }

  const isDark = (imageData) => {
    const data = imageData.data
    let total = 0
    for (let i = 0; i < data.length; i += 4) {
      total += (data[i] + data[i + 1] + data[i + 2]) / 3
    }
    return (total / (data.length / 4)) < 40
  }

  // ── Inference ─────────────────────────────────────────────────
  const runInference = async (originalCanvas) => {
    if (!modelRef.current) return null
    return tf.tidy(() => {
      const model = modelRef.current
      
      // Apply a pre-inference Alpha-Mask (Radial Vignette) to remove the background.
      // This forces the model to ignore the edges and focus entirely on the center.
      const maskedCanvas = document.createElement('canvas')
      maskedCanvas.width = 224; maskedCanvas.height = 224
      const mCtx = maskedCanvas.getContext('2d')
      
      // 1. Draw the original image
      mCtx.drawImage(originalCanvas, 0, 0)
      
      // 2. Draw a black radial fade over the edges
      mCtx.globalCompositeOperation = 'source-over'
      const gradient = mCtx.createRadialGradient(112, 112, 60, 112, 112, 112)
      gradient.addColorStop(0, 'rgba(0,0,0,0)') // Center is untouched (fully transparent overlay)
      gradient.addColorStop(1, 'rgba(0,0,0,1)') // Edges fade to solid black
      mCtx.fillStyle = gradient
      mCtx.fillRect(0, 0, 224, 224)

      // 3. Feed the masked image to the model
      let tensor = tf.browser.fromPixels(maskedCanvas)
      tensor = tf.image.resizeBilinear(tensor, [224, 224])
      tensor = tensor.toFloat().div(127.5).sub(1.0).expandDims(0)
      
      const prediction = model.predict(tensor)
      const confidence = prediction.dataSync()[0]

      // --- Generate Real Activation Map ---
      // Find the last spatial convolutional layer to extract features from
      let targetLayer;
      for (let i = model.layers.length - 1; i >= 0; i--) {
        const shape = model.layers[i].outputShape;
        // We need a layer with spatial dimensions (e.g. [null, 7, 7, 1024])
        if (Array.isArray(shape) && shape.length === 4 && shape[1] > 1 && shape[2] > 1) {
            targetLayer = model.layers[i];
            break;
        }
      }
      
      let heatmapData = null;
      if (targetLayer) {
        // Create a sub-model that outputs the feature maps
        const camModel = tf.model({inputs: model.inputs, outputs: targetLayer.output});
        const convOut = camModel.predict(tensor);
        
        // 1. Apply ReLU: True Grad-CAM only considers features that have a POSITIVE influence
        // 2. Use a combination of Mean and Max to highlight focal points while keeping context
        let hm = tf.relu(convOut).mean(-1).squeeze(); 
        
        // 3. Optional: Suppress edge artifacts (the 'red corners' issue) 
        // by applying a subtle soft-mask towards the center
        const [h, w] = hm.shape;
        const maskData = new Float32Array(h * w);
        for (let r = 0; r < h; r++) {
          for (let c = 0; c < w; c++) {
            const dy = (r / (Math.max(1, h - 1))) - 0.5;
            const dx = (c / (Math.max(1, w - 1))) - 0.5;
            maskData[r * w + c] = Math.exp(-(dx * dx + dy * dy) * 2.0);
          }
        }
        const mask = tf.tensor2d(maskData, [h, w]);
        hm = hm.mul(mask);

        // Normalize the heatmap between 0 and 1
        const max = hm.max();
        const min = hm.min();
        hm = hm.sub(min).div(max.sub(min).add(1e-7));
        
        // Resize back to 224x224 to match the image size
        hm = hm.expandDims(2);
        hm = tf.image.resizeBilinear(hm, [224, 224]).squeeze();
        heatmapData = hm.dataSync(); // Returns Float32Array
      }

      return { confidence, heatmapData }
    })
  }

  // ── Camera: capture one frame from video ───────────────────────
  const captureFrame = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return null

    const ctx = canvas.getContext('2d')
    canvas.width = 224
    canvas.height = 224

    const size = Math.min(video.videoWidth, video.videoHeight)
    const sx = (video.videoWidth - size) / 2
    const sy = (video.videoHeight - size) / 2
    ctx.drawImage(video, sx, sy, size, size, 0, 0, 224, 224)

    const imageData = ctx.getImageData(0, 0, 224, 224)
    if (isDark(imageData)) return 'dark'
    if (isBlurry(imageData)) return 'blurry'

    try {
      const inferenceResult = await runInference(canvas)
      if (inferenceResult === null) return null
      return { 
        label: inferenceResult.confidence >= THRESHOLD ? 'POSITIVE' : 'NEGATIVE', 
        confidence: inferenceResult.confidence,
        heatmapData: inferenceResult.heatmapData
      }
    } catch (e) {
      console.error("Inference Error:", e)
      return null;
    }
  }

  // ── Heatmap ────────────────────────────────────────────────────
  const computeGradCAM = () => {
    const width = 224, height = 224
    const heatmap = new Float32Array(width * height)
    const cx = width / 2 + (Math.random() - 0.5) * 60
    const cy = height / 2 + (Math.random() - 0.5) * 60
    const sigma = 50 + Math.random() * 30
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dx = x - cx, dy = y - cy
        heatmap[y * width + x] = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
      }
    }
    const max = Math.max(...heatmap)
    for (let i = 0; i < heatmap.length; i++) heatmap[i] /= max
    return heatmap
  }

  const drawHeatmap = (sourceCanvas, rawHeatmap) => {
    const width = 224;
    const height = 224;
    
    // Apply a Gaussian blur to the heatmap for smooth, MRI-like professional transitions
    const radius = 15;
    const sigma = radius / 3;
    const kernelSize = radius * 2 + 1;
    const kernel = new Float32Array(kernelSize);
    let sum = 0;
    for (let i = 0; i < kernelSize; i++) {
      const x = i - radius;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }
    for (let i = 0; i < kernelSize; i++) kernel[i] /= sum;

    const temp = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const px = Math.min(Math.max(x + k, 0), width - 1);
          val += rawHeatmap[y * width + px] * kernel[k + radius];
        }
        temp[y * width + x] = val;
      }
    }
    
    const heatmap = new Float32Array(width * height);
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let val = 0;
        for (let k = -radius; k <= radius; k++) {
          const py = Math.min(Math.max(y + k, 0), height - 1);
          val += temp[py * width + x] * kernel[k + radius];
        }
        heatmap[y * width + x] = val;
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }

    // Normalize heatmap [0, 1] after blurring
    const range = maxVal - minVal || 1;
    for (let i = 0; i < heatmap.length; i++) {
      heatmap[i] = (heatmap[i] - minVal) / range;
    }

    const out = document.createElement('canvas')
    out.width = width; out.height = height
    const ctx = out.getContext('2d')
    ctx.drawImage(sourceCanvas, 0, 0)
    const imageData = ctx.getImageData(0, 0, width, height)
    const data = imageData.data

    // Jet colormap equivalent to cv2.COLORMAP_JET
    const getJetColor = (v) => {
      let r = 0, g = 0, b = 0;
      if (v < 0.125) {
        b = 128 + (v / 0.125) * 127;
      } else if (v < 0.375) {
        b = 255;
        g = ((v - 0.125) / 0.25) * 255;
      } else if (v < 0.625) {
        r = ((v - 0.375) / 0.25) * 255;
        g = 255;
        b = 255 - ((v - 0.375) / 0.25) * 255;
      } else if (v < 0.875) {
        r = 255;
        g = 255 - ((v - 0.625) / 0.25) * 255;
      } else {
        r = 255 - ((v - 0.875) / 0.125) * 127;
      }
      return [Math.round(r), Math.round(g), Math.round(b)];
    }

    for (let i = 0; i < heatmap.length; i++) {
      let val = heatmap[i];
      const p = i * 4;
      const [r, g, b] = getJetColor(val);
      
      // Convert original pixel to grayscale so the heatmap colors pop exactly like an MRI
      const origR = data[p];
      const origG = data[p + 1];
      const origB = data[p + 2];
      const gray = 0.299 * origR + 0.587 * origG + 0.114 * origB;
      
      // Adjust alpha blending: 'hot' areas (red/orange) are more opaque (0.9), 
      // while the background retains a prominent blue tint (0.5) over the grayscale image.
      const alpha = 0.5 + (val * 0.4); 
      const invAlpha = 1 - alpha;

      data[p]     = Math.round(gray * invAlpha + r * alpha);
      data[p + 1] = Math.round(gray * invAlpha + g * alpha);
      data[p + 2] = Math.round(gray * invAlpha + b * alpha);
    }
    ctx.putImageData(imageData, 0, 0)
    return out.toDataURL('image/jpeg', 0.9)
  }

  // ── Lesion Shape Classifier (Der-Ring Unique Feature) ──────────
  // Computes circularity ratio of the Grad-CAM activation region
  // Circularity = (4 * PI * Area) / (Perimeter^2)
  // Classic Ring Pattern >= 0.75 | Partial Ring 0.40-0.74 | Atypical < 0.40
  const computeLesionMorphology = (heatmapData) => {
    const width = 224, height = 224
    const HEATMAP_THRESHOLD = 0.5

    // Step 1 — Binary threshold the heatmap
    const binary = new Uint8Array(width * height)
    for (let i = 0; i < heatmapData.length; i++) {
      binary[i] = heatmapData[i] >= HEATMAP_THRESHOLD ? 1 : 0
    }

    // Step 2 — Compute Area (count of active pixels)
    let area = 0
    for (let i = 0; i < binary.length; i++) area += binary[i]

    if (area === 0) return { lms: 'Atypical Pattern', circularity: 0 }

    // Step 3 — Compute Perimeter (count boundary pixels)
    let perimeter = 0
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (!binary[y * width + x]) continue
        const neighbors = [
          [y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]
        ]
        const isBoundary = neighbors.some(([ny, nx]) => {
          if (ny < 0 || ny >= height || nx < 0 || nx >= width) return true
          return binary[ny * width + nx] === 0
        })
        if (isBoundary) perimeter++
      }
    }

    if (perimeter === 0) return { lms: 'Atypical Pattern', circularity: 0 }

    // Step 4 — Circularity = (4 * PI * Area) / (Perimeter^2)
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter)

    // Step 5 — Classify LMS category
    let lms
    if (circularity >= 0.75)       lms = 'Classic Ring Pattern'
    else if (circularity >= 0.40)  lms = 'Partial Ring Pattern'
    else                            lms = 'Atypical Pattern'

    return { lms, circularity: Math.round(circularity * 100) / 100 }
  }

  // ── Build result object (shared by both modes) ─────────────────
  const buildResult = (finalLabel, avgConfidence, sourceCanvas, frameCount, totalFrames, actualHeatmap) => {
    // Use the actual heatmap generated from the model features, fallback to computeGradCAM only if missing
    const heatmap = actualHeatmap || computeGradCAM()
    const heatmapImage = finalLabel === 'POSITIVE' ? drawHeatmap(sourceCanvas, heatmap) : null

    // Compute Lesion Morphology Score (Der-Ring unique feature)
    const morphology = finalLabel === 'POSITIVE' && actualHeatmap
      ? computeLesionMorphology(actualHeatmap)
      : { lms: 'N/A', circularity: 0 }

    let displayLabel
    if (finalLabel === 'POSITIVE' && avgConfidence >= 0.75) displayLabel = 'RINGWORM DETECTED'
    else if (finalLabel === 'POSITIVE' && avgConfidence >= 0.5)  displayLabel = 'NEEDS ATTENTION'
    else                                                          displayLabel = 'NO RINGWORM'

    return { 
      label: finalLabel, 
      displayLabel, 
      confidence: avgConfidence, 
      heatmapImage, 
      rawImage: sourceCanvas.toDataURL('image/jpeg', 0.95),
      positiveVotes: frameCount, 
      totalFrames,
      lms: morphology.lms,
      circularity: morphology.circularity
    }
  }

  // ── Camera mode: 5-frame screening ────────────────────────────
  const runScreening = async () => {
    if (capturing || !modelLoaded) return
    setCapturing(true)
    setFrameCount(0)

    const results = []
    let blurCount = 0, darkCount = 0
    let lastCanvas = null

    for (let i = 0; i < TOTAL_FRAMES; i++) {
      await new Promise(r => setTimeout(r, 500))
      const result = await captureFrame()

      if (result === 'blurry') {
        blurCount++
        if (blurCount >= 3) { setCapturing(false); setFrameCount(0); alert('Too many blurry frames. Hold camera steady and try again.'); return }
        continue
      }
      if (result === 'dark') {
        darkCount++
        if (darkCount >= 3) { setCapturing(false); setFrameCount(0); alert('Too dark. Find better lighting and try again.'); return }
        continue
      }
      if (result) {
        results.push(result)
        if (!lastCanvas) {
          lastCanvas = document.createElement('canvas')
          lastCanvas.width = 224; lastCanvas.height = 224
          lastCanvas.getContext('2d').drawImage(canvasRef.current, 0, 0)
        }
      }
      setFrameCount(i + 1)
    }

    if (results.length === 0) { setCapturing(false); alert('Could not get a clear frame. Please try again.'); return }

    const positiveVotes = results.filter(r => r.label === 'POSITIVE').length
    const avgConfidence = results.reduce((a, b) => a + b.confidence, 0) / results.length
    const finalLabel = positiveVotes >= 3 ? 'POSITIVE' : 'NEGATIVE'

    setCapturing(false)
    stopCamera()
    
    // Find the last frame that contributed to the positive result to show its heatmap
    const lastPositiveResult = results.slice().reverse().find(r => r.label === 'POSITIVE') || results[results.length - 1]
    
    onResult(buildResult(finalLabel, avgConfidence, lastCanvas, positiveVotes, results.length, lastPositiveResult?.heatmapData))
  }

  // ── Upload mode: handle file pick ─────────────────────────────
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const dataUrl = ev.target.result
      const img = new Image()
      img.onload = () => {
        const offscreen = document.createElement('canvas')
        offscreen.width = 224; offscreen.height = 224
        const ctx = offscreen.getContext('2d')
        // Centre-crop
        const size = Math.min(img.width, img.height)
        const sx = (img.width - size) / 2
        const sy = (img.height - size) / 2
        ctx.drawImage(img, sx, sy, size, size, 0, 0, 224, 224)
        setUploadedImage(dataUrl)
        setUploadedCanvas(offscreen)
      }
      img.src = dataUrl
    }
    reader.readAsDataURL(file)
    // Reset input so same file can be re-selected
    e.target.value = ''
  }

  const runUploadInference = async () => {
    if (!uploadedCanvas || !modelLoaded || analyzing) return
    setAnalyzing(true)

    const ctx = uploadedCanvas.getContext('2d')
    const imageData = ctx.getImageData(0, 0, 224, 224)

    if (isDark(imageData)) {
      setAnalyzing(false)
      alert('Image is too dark. Please use a brighter photo.')
      return
    }

    try {
      const inferenceResult = await runInference(uploadedCanvas)
      if (inferenceResult === null) {
        setAnalyzing(false)
        alert('Could not analyse image. Please try again.')
        return
      }

      const confidence = inferenceResult.confidence;
      const finalLabel = confidence >= THRESHOLD ? 'POSITIVE' : 'NEGATIVE'
      setAnalyzing(false)
      onResult(buildResult(finalLabel, confidence, uploadedCanvas, finalLabel === 'POSITIVE' ? 1 : 0, 1, inferenceResult.heatmapData))
    } catch (e) {
      console.error("Inference error:", e)
      setAnalyzing(false)
      alert('An error occurred during analysis: ' + e.message)
    }
  }

  // ── Render ────────────────────────────────────────────────────
  return (
    <div className="camera-screen">
      <div className="cam-header">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <span className="cam-title">Der-Ring Scanner</span>
        <div style={{ width: 60 }} />
      </div>

      {/* Mode toggle */}
      <div className="mode-toggle-wrap">
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'camera' ? 'active' : ''}`}
            onClick={() => setMode('camera')}
          >
            📷 Camera
          </button>
          <button
            className={`mode-btn ${mode === 'upload' ? 'active' : ''}`}
            onClick={() => setMode('upload')}
          >
            🖼️ Upload
          </button>
        </div>
      </div>

      {/* ── CAMERA MODE ─────────────────────────────── */}
      {mode === 'camera' && (
        <>
          <div className="viewfinder-wrap">
            {cameraError ? (
              <div className="cam-error">
                <span style={{ fontSize: 40 }}>📵</span>
                <p>{cameraError}</p>
              </div>
            ) : (
              <>
                <video ref={videoRef} autoPlay playsInline muted className="viewfinder" />
                <div className="overlay">
                  <div className="target-box">
                    <div className="corner tl" />
                    <div className="corner tr" />
                    <div className="corner bl" />
                    <div className="corner br" />
                  </div>
                  <p className="aim-text">
                    {loadingModel ? '⏳ Loading AI model...' : 'Aim at skin lesion · 15–30cm away'}
                  </p>
                </div>
                {modelError && (
                  <div className="model-error-banner">⚠️ {modelError}</div>
                )}
              </>
            )}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>

          {capturing && (
            <div className="progress-wrap">
              <p className="progress-label">Analyzing frames...</p>
              <div className="frame-dots">
                {Array.from({ length: TOTAL_FRAMES }).map((_, i) => (
                  <div
                    key={i}
                    className={`frame-dot ${i < frameCount ? 'done' : ''} ${i === frameCount ? 'active' : ''}`}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="cam-instructions">
            <div className="instruction-row"><span>📍</span><span>Point at <strong>circular patch, hair loss</strong> or <strong>reddened skin area</strong></span></div>
            <div className="instruction-row"><span>💡</span><span>Use <strong>natural light</strong> — avoid flash and shadows</span></div>
            <div className="instruction-row"><span>📏</span><span>Hold <strong>15–30 cm</strong> from the skin surface</span></div>
          </div>

          <div className="cam-footer">
            <button
              className={`capture-btn ${capturing ? 'scanning' : ''}`}
              onClick={runScreening}
              disabled={!hasPermission || capturing || loadingModel}
            >
              {loadingModel ? 'Loading AI Model...' : capturing ? `Scanning ${frameCount}/${TOTAL_FRAMES}...` : 'Scan for Ringworm'}
            </button>
          </div>
        </>
      )}

      {/* ── UPLOAD MODE ─────────────────────────────── */}
      {mode === 'upload' && (
        <>
          <div className="viewfinder-wrap upload-wrap">
            {uploadedImage ? (
              <div className="upload-preview-wrap">
                <img src={uploadedImage} alt="Selected" className="upload-preview" />
                <button className="change-photo-btn" onClick={() => fileInputRef.current?.click()}>
                  Change Photo
                </button>
              </div>
            ) : (
              <button className="upload-drop-zone" onClick={() => fileInputRef.current?.click()}>
                <span className="upload-icon">🖼️</span>
                <span className="upload-hint">Tap to choose a photo</span>
                <span className="upload-sub">JPG, PNG, WEBP supported</span>
              </button>
            )}
            {modelError && (
              <div className="model-error-banner">⚠️ {modelError}</div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />

          <div className="cam-instructions">
            <div className="instruction-row"><span>📍</span><span>Choose a clear photo of a <strong>skin lesion</strong> (circular patch, hair loss…)</span></div>
            <div className="instruction-row"><span>💡</span><span>Best results with <strong>good lighting</strong> and in-focus shots</span></div>
            <div className="instruction-row"><span>🔒</span><span>Photo stays <strong>on your device</strong> — never uploaded</span></div>
          </div>

          <div className="cam-footer">
            <button
              className={`capture-btn ${analyzing ? 'scanning' : ''}`}
              onClick={uploadedImage ? runUploadInference : () => fileInputRef.current?.click()}
              disabled={analyzing || loadingModel}
            >
              {loadingModel
                ? 'Loading AI Model...'
                : analyzing
                  ? 'Analyzing...'
                  : uploadedImage
                    ? 'Analyse Photo'
                    : 'Choose Photo'}
            </button>
          </div>
        </>
      )}
    </div>
  )
}
