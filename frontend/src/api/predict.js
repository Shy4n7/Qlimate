const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

const CLASS_NAMES = ['Normal','Drought','Wet_Flood','Heat_Extreme','Cold_Extreme']

const MOCK_STATES = [
  'Andaman and Nicobar','Andhra Pradesh','Arunachal Pradesh','Assam','Bihar',
  'Chhattisgarh','Gujarat','Haryana','Himachal Pradesh','Jammu and Kashmir',
  'Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra','Manipur',
  'Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim',
  'Tamil Nadu','Telangana','Tripura','Uttar Pradesh','Uttarakhand','West Bengal',
]

function mockPredict(state, month, year) {
  // Deterministic mock based on (state, month, year) — UI always works even without backend
  const hash = [...state].reduce((a, c) => a + c.charCodeAt(0), month * 31 + (year || 0))
  const cIdx = hash % CLASS_NAMES.length
  const qIdx = (hash + 2) % CLASS_NAMES.length
  const cConf = 0.45 + (hash % 40) / 100
  const qConf = 0.25 + (hash % 30) / 100

  const classicalLabel = CLASS_NAMES[cIdx]
  const quantumLabel   = CLASS_NAMES[qIdx]
  const agreement      = classicalLabel === quantumLabel

  return {
    state,
    month,
    historical_label: classicalLabel,
    classical: {
      label:         classicalLabel,
      label_display: _display(classicalLabel),
      confidence:    cConf,
      all_proba:     _uniform(cIdx, cConf),
      model:         'XGBoost',
      n_train:       6958,
      n_features:    16,
    },
    quantum: {
      label:            quantumLabel,
      label_display:    _display(quantumLabel),
      confidence:       qConf,
      all_proba:        _uniform(qIdx, qConf),
      model:            'QSVC',
      n_train:          400,
      n_features:       4,
      confidence_note:  'Estimated from SVM decision scores via softmax — not a calibrated probability',
    },
    agreement,
    confidence_gap:    Math.abs(cConf - qConf),
    difference_reason: agreement
      ? 'Both models agree on this state-month pattern (mock data — start the backend for real predictions).'
      : 'Models disagree here (mock data — start the backend for real predictions).',
    is_mock: true,
    is_projection: (year || 0) > 2024,
  }
}

function _display(label) {
  const map = {
    Normal: 'Normal', Drought: 'Drought', Wet_Flood: 'Flood / Wet',
    Heat_Extreme: 'Heat Extreme', Cold_Extreme: 'Cold Extreme',
  }
  return map[label] || label
}

function _uniform(winnerIdx, winnerConf) {
  const rest = (1 - winnerConf) / (CLASS_NAMES.length - 1)
  return Object.fromEntries(CLASS_NAMES.map((c, i) => [c, i === winnerIdx ? winnerConf : rest]))
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

export { MONTHS, CLASS_NAMES }
