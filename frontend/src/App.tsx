import { useState, useRef, useCallback } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000/api/v1/analyze'

interface Prediction {
  country: string
  region: string | null
  direction: string | null
  confidence: number
  country_probabilities: Record<string, number>
}

interface Explanation {
  top_clues: Array<{
    feature: string
    countries: string[]
    weight: number
  }>
  driving_side: string
  script_type: string
  vegetation: string
  terrain: string
  architecture_style: string
  road_type: string
  ocr_location_hint: string | null
}

interface AnalysisResult {
  session_id: string
  prediction: Prediction
  explanation: Explanation
}

export default function App() {
  const [image, setImage] = useState<string | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File) => {
    setImageFile(file)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setImage(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) handleFile(file)
  }, [])

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(true)
  }

  const onDragLeave = () => setDragging(false)

  const analyze = async () => {
    if (!imageFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', imageFile)

    try {
      const response = await axios.post<AnalysisResult>(API_URL, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(response.data)
    } catch (err) {
      setError('Analysis failed. Make sure the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  const pct = (n: number) => `${Math.round(n * 100)}%`

  return (
    <div className="app">
      <div className="header">
        <h1>GeoGuessr AI</h1>
        <p>Upload a Street View screenshot to predict the location</p>
      </div>

      {!image ? (
        <div
          className={`upload-zone ${dragging ? 'dragging' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div className="upload-icon">📍</div>
          <h3>Drop a Street View screenshot here</h3>
          <p>or click to browse — JPEG, PNG, WebP supported</p>
        </div>
      ) : (
        <div className="preview-container">
          <img src={image} alt="Preview" />
          <button className="change-btn" onClick={() => {
            setImage(null)
            setImageFile(null)
            setResult(null)
            setError(null)
          }}>
            Change image
          </button>
        </div>
      )}

      {image && (
        <button
          className="analyze-btn"
          onClick={analyze}
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze Image'}
        </button>
      )}

      {error && <div className="error">{error}</div>}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Running AI analysis...</p>
        </div>
      )}

      {result && (
        <div className="results">

          {/* Main prediction */}
          <div className="card">
            <h2>Prediction</h2>
            <div className="prediction-country">{result.prediction.country}</div>
            <div className="prediction-region">
              {result.prediction.region
                ? `${result.prediction.region} region`
                : 'Region unknown'}
              {result.prediction.direction
                ? ` · ${result.prediction.direction}`
                : ''}
            </div>
            <div className="confidence-label">Confidence</div>
            <div className="confidence-bar-bg">
              <div
                className="confidence-bar-fill"
                style={{ width: pct(result.prediction.confidence) }}
              />
            </div>
            <div className="confidence-pct">{pct(result.prediction.confidence)}</div>
          </div>

          {/* Country probabilities */}
          <div className="card">
            <h2>Top Countries</h2>
            {Object.entries(result.prediction.country_probabilities).map(([country, prob]) => (
              <div className="prob-row" key={country}>
                <div className="prob-country">{country}</div>
                <div className="prob-bar-bg">
                  <div
                    className="prob-bar-fill"
                    style={{ width: pct(prob) }}
                  />
                </div>
                <div className="prob-pct">{pct(prob)}</div>
              </div>
            ))}
          </div>

          {/* Top clues */}
          <div className="card full-width">
            <h2>Why this location?</h2>
            {result.explanation.top_clues.map((clue, i) => (
              <div className="clue-row" key={i}>
                <div className="clue-feature">{clue.feature}</div>
                <div className="clue-countries">→ {clue.countries.join(', ')}</div>
                <div
                  className="clue-weight-bar"
                  style={{ width: `${Math.round(clue.weight * 100)}%` }}
                />
              </div>
            ))}
          </div>

          {/* Detected features */}
          <div className="card full-width">
            <h2>Detected Features</h2>
            <div className="features-grid">
              <div className="feature-item">
                <div className="feature-label">Driving side</div>
                <div className="feature-value">{result.explanation.driving_side}</div>
              </div>
              <div className="feature-item">
                <div className="feature-label">Script type</div>
                <div className="feature-value">{result.explanation.script_type || 'unknown'}</div>
              </div>
              <div className="feature-item">
                <div className="feature-label">Vegetation</div>
                <div className="feature-value">{result.explanation.vegetation}</div>
              </div>
              <div className="feature-item">
                <div className="feature-label">Terrain</div>
                <div className="feature-value">{result.explanation.terrain}</div>
              </div>
              <div className="feature-item">
                <div className="feature-label">Architecture</div>
                <div className="feature-value">{result.explanation.architecture_style}</div>
              </div>
              <div className="feature-item">
                <div className="feature-label">Road type</div>
                <div className="feature-value">{result.explanation.road_type}</div>
              </div>
              {result.explanation.ocr_location_hint && (
                <div className="feature-item full-width">
                  <div className="feature-label">Location hint (OCR)</div>
                  <div className="feature-value">{result.explanation.ocr_location_hint}</div>
                </div>
              )}
            </div>
          </div>

        </div>
      )}
    </div>
  )
}