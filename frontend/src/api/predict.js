const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

const MOCK_STATES = [
  'Andaman and Nicobar','Andhra Pradesh','Arunachal Pradesh','Assam','Bihar',
  'Chhattisgarh','Gujarat','Haryana','Himachal Pradesh','Jammu and Kashmir',
  'Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra','Manipur',
  'Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim',
  'Tamil Nadu','Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal',
]

// Base temperatures (°C) per month — approximate India-wide seasonal pattern
const MONTH_BASE_TEMPS = [18, 20, 25, 30, 34, 33, 30, 29, 29, 27, 23, 19]

// State offset — hotter states (Rajasthan, Gujarat) vs cooler (Himachal, Arunachal)
function _stateOffset(state) {
  const hash = [...state].reduce((a, c) => a + c.charCodeAt(0), 0)
  // Range: -8 to +8 °C offset from base
  return ((hash % 17) - 8)
}

// Deterministic pseudo-random float in [min, max] based on a seed integer
function _seededFloat(seed, min, max) {
  // Simple LCG-style hash
  const h = ((seed * 1664525 + 1013904223) >>> 0) / 4294967296
  return min + h * (max - min)
}

function mockPredict(state, month, year) {
  // Deterministic mock based on (state, month, year) — UI always works even without backend
  const monthIdx = (month - 1 + 12) % 12  // 0-based
  const baseTemp = MONTH_BASE_TEMPS[monthIdx]
  const stateOff = _stateOffset(state)

  // Seed combines state hash, month, and year for determinism
  const stateHash = [...state].reduce((a, c) => a + c.charCodeAt(0), 0)
  const seed = stateHash * 31 + month * 7 + (year || 2024)

  // Classical: XGBoost Regressor
  const classicalTemp = parseFloat((baseTemp + stateOff + _seededFloat(seed, -2, 2)).toFixed(1))

  // Quantum: QSVR — slight divergence from classical, grows for future years
  const yearOffset = year > 2024 ? (year - 2024) * 0.05 : 0
  const quantumTemp = parseFloat((classicalTemp + _seededFloat(seed + 1, -1.5, 1.5) + yearOffset).toFixed(1))

  const tempDelta = parseFloat((quantumTemp - classicalTemp).toFixed(2))
  const absDelta = Math.abs(tempDelta)
  const direction = tempDelta >= 0 ? 'warmer' : 'cooler'
  const divergenceNote = absDelta < 0.3
    ? 'Classical and quantum models are in close agreement for this state-month combination (mock data — start the backend for real predictions).'
    : `Quantum model predicts ${absDelta.toFixed(1)} °C ${direction} than classical (mock data — start the backend for real predictions).`

  return {
    state,
    month,
    year: year || 2024,
    is_future: (year || 2024) > 2024,
    classical: {
      model:            'XGBoost Regressor',
      predicted_temp_c: classicalTemp,
      n_train:          6958,
      n_features:       17,
    },
    quantum: {
      model:            'QSVR',
      predicted_temp_c: quantumTemp,
      n_train:          400,
      n_features:       4,
    },
    temp_delta_c:    tempDelta,
    divergence_note: divergenceNote,
    is_mock: true,
  }
}

function _mockForecastTemps(state, month, baseClassical) {
  const stateHash = [...state].reduce((a, c) => a + c.charCodeAt(0), 0)
  const years = Array.from({ length: 11 }, (_, i) => 2025 + i)

  const classicalTemps = years.map(year => {
    const seed = stateHash * 31 + month * 7 + year
    const trend = (year - 2025) * 0.04  // slight warming trend
    return parseFloat((baseClassical + trend + _seededFloat(seed, -0.5, 0.5)).toFixed(1))
  })

  const quantumTemps = years.map((year, i) => {
    const seed = stateHash * 31 + month * 7 + year + 1
    const divergence = (year - 2025) * 0.05  // quantum diverges slightly more
    return parseFloat((classicalTemps[i] + divergence + _seededFloat(seed, -0.8, 0.8)).toFixed(1))
  })

  return { years, classicalTemps, quantumTemps }
}

export async function predict(state, month, year) {
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state, month, year }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch {
    return mockPredict(state, month, year)
  }
}

export async function fetchStates() {
  try {
    const res = await fetch('/api/states')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    return data.states
  } catch {
    return MOCK_STATES
  }
}

export async function fetchForecast(state, month) {
  try {
    const params = new URLSearchParams({ state, month: String(month) })
    const res = await fetch(`/api/forecast?${params}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch {
    // Fallback mock forecast: compute a plausible base temp for this state+month
    const monthIdx = (month - 1 + 12) % 12
    const baseTemp = MONTH_BASE_TEMPS[monthIdx] + _stateOffset(state)
    const { years, classicalTemps, quantumTemps } = _mockForecastTemps(state, month, baseTemp)
    return {
      state,
      month,
      years,
      classical_temps: classicalTemps,
      quantum_temps:   quantumTemps,
      is_mock: true,
    }
  }
}

export { MONTHS }
