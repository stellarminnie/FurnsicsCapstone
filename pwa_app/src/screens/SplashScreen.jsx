import { useEffect, useState } from 'react'
import './SplashScreen.css'

export default function SplashScreen({ onStart }) {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 100)
    return () => clearTimeout(t)
  }, [])

  const features = [
    { label: 'AI Detection',        desc: 'MobileNetV3 on-device' },
    { label: 'Grad-CAM Heatmaps',   desc: 'Visual evidence per result' },
    { label: 'Lesion Shape Classifier', desc: 'Circularity-based ring pattern analysis' },
    { label: 'Works Offline',       desc: 'No internet required' },
  ]

  return (
    <div className={`splash ${ready ? 'ready' : ''}`}>

      <div className="splash-top">
        <div className="splash-badge">Edge-AI · Offline-First</div>
      </div>

      <div className="splash-hero">
        <div className="logo-mark">
          <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
            {/* Circular ring shape representing dermatophyte lesion */}
            <circle cx="18" cy="18" r="14" stroke="white" strokeWidth="2.5" fill="none"/>
            <circle cx="18" cy="18" r="8" stroke="white" strokeWidth="1.5" fill="none" strokeDasharray="3 2"/>
            <circle cx="18" cy="18" r="3" fill="white"/>
          </svg>
        </div>
        <h1 className="splash-title">Der-Ring</h1>
        <p className="splash-sub">Canine Ringworm Screening</p>
      </div>

      <div className="splash-features">
        {features.map((f) => (
          <div className="feature-row" key={f.label}>
            <div className="feature-dot" />
            <div>
              <div className="feature-label">{f.label}</div>
              <div className="feature-desc">{f.desc}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="splash-footer">
        <button className="btn-primary" onClick={onStart}>
          Begin Screening
        </button>
        <p className="splash-disclaimer">Triage use only · Not a veterinary diagnosis</p>
      </div>

    </div>
  )
}
