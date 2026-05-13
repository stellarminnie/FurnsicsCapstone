import { useState } from 'react'
import './ResultScreen.css'

export default function ResultScreen({ result, onScanAgain, onHome }) {
  const [peeking, setPeeking] = useState(false)
  
  const isPositive = result?.label === 'POSITIVE'
  const isAttention = result?.displayLabel === 'NEEDS ATTENTION'
  const confidence = Math.round((result?.confidence || 0) * 100)

  let actionText = 'Continue regular monitoring'
  let actionColor = '#1E8449'
  if (result?.displayLabel === 'RINGWORM DETECTED') {
    actionText = 'Urgent veterinary referral required immediately'
    actionColor = '#C0392B'
  } else if (result?.displayLabel === 'NEEDS ATTENTION') {
    actionText = 'Veterinary consultation recommended within 24 hours'
    actionColor = '#E67E22'
  }

  const headerColor = isPositive
    ? (isAttention ? 'var(--warning)' : 'var(--danger)')
    : 'var(--success)'

  // LMS color based on category
  const lmsColor = result?.lms === 'Classic Ring Pattern' ? '#C0392B'
    : result?.lms === 'Partial Ring Pattern' ? '#E67E22'
    : '#7F8C8D'

  const lmsDescription = result?.lms === 'Classic Ring Pattern'
    ? 'Activation shape strongly matches dermatophyte ring morphology — high-confidence ringworm.'
    : result?.lms === 'Partial Ring Pattern'
    ? 'Partial circular pattern detected — possible early or healing ringworm. Vet consultation recommended.'
    : 'Shape does not match ring pattern — atypical or non-ringworm condition. Vet verification required.'

  return (
    <div className="result-screen">

      {/* Header */}
      <div className="result-header">
        <div className="result-icon" style={{ color: headerColor }}>
          {result?.displayLabel === 'RINGWORM DETECTED' && '⚠️'}
          {result?.displayLabel === 'NEEDS ATTENTION' && '🔶'}
          {result?.displayLabel === 'NO RINGWORM' && '✅'}
        </div>
        <h1 className="result-label" style={{ color: headerColor }}>{result?.displayLabel}</h1>
        <p className="result-confidence">
          AI Confidence: {confidence}%
        </p>
        <p className="result-votes">
          {result?.positiveVotes}/{result?.totalFrames} frames flagged positive
        </p>
      </div>

      <div className="result-body">

        {/* Heatmap */}
        {result?.heatmapImage && (
          <div className="result-section">
            <div className="section-header-row">
              <div className="section-title">Grad-CAM Activation Map</div>
              <button 
                className={`peek-btn ${peeking ? 'active' : ''}`}
                onMouseDown={() => setPeeking(true)}
                onMouseUp={() => setPeeking(false)}
                onMouseLeave={() => setPeeking(false)}
                onTouchStart={() => setPeeking(true)}
                onTouchEnd={() => setPeeking(false)}
              >
                👁️ Peek Original
              </button>
            </div>
            
            <div className="heatmap-wrap">
              <img
                src={peeking ? result.rawImage : result.heatmapImage}
                alt="Diagnosis view"
                className="heatmap-img"
              />
              <p className="heatmap-caption">
                {peeking 
                  ? 'Viewing original raw image' 
                  : 'Red areas = regions that influenced the detection'}
              </p>
            </div>
          </div>
        )}

        {/* Lesion Shape Classifier — Der-Ring Unique Feature */}
        {result?.lms && result.lms !== 'N/A' && (
          <div className="result-section">
            <div className="section-title">Lesion Morphology Score (LMS)</div>
            <div className="action-box" style={{ borderColor: lmsColor }}>
              <p className="action-text" style={{ color: lmsColor }}>
                {result.lms}
              </p>
              <p style={{ fontSize: '0.82rem', color: '#555', marginTop: 4 }}>
                Circularity: <strong>{result.circularity}</strong>
              </p>
              <p className="action-note">{lmsDescription}</p>
            </div>
          </div>
        )}

        {/* Action guideline */}
        <div className="result-section">
          <div className="section-title">Recommended Action</div>
          <div className="action-box" style={{ borderColor: actionColor }}>
            <p className="action-text">{actionText}</p>
            {isPositive && (
              <p className="action-note">
                This is a preliminary screening result only.
                A licensed veterinarian must confirm the diagnosis.
              </p>
            )}
            {!isPositive && (
              <p className="action-note">
                No ringworm indicators detected. Continue regular
                monitoring and consult a vet if symptoms develop.
              </p>
            )}
          </div>
        </div>

        {/* Disclaimer */}
        <div className="result-disclaimer">
          Der-Ring is a triage aid only · Not a veterinary diagnosis
        </div>

        {/* Action buttons */}
        <div className="result-actions">
          <button className="btn-primary" onClick={onScanAgain}>
            Scan Another Dog
          </button>
          <button className="btn-secondary" onClick={onHome}>
            Home
          </button>
        </div>

      </div>
    </div>
  )
}
