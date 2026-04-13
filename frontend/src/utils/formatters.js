export const fmt2 = (v) => (v == null ? 'N/A' : v.toFixed(2))
export const fmt3 = (v) => (v == null ? 'N/A' : v.toFixed(3))
export const fmtPct = (v) => (v == null ? 'N/A' : `${(v * 100).toFixed(1)}%`)
export const fmtMin = (s) => (s == null ? 'N/A' : `${(s / 60).toFixed(1)} min`)
export const fmtSec = (s) => (s == null ? 'N/A' : `${s.toFixed(1)}s`)

export const MODEL_COLORS = {
  random_forest:  '#3b82f6',
  svm:            '#0ea5e9',
  xgboost:        '#06b6d4',
  neural_network: '#22d3ee',
  QSVC:           '#7c3aed',
  VQC:            '#a855f7',
}

export const MODEL_LABELS = {
  random_forest:  'Random Forest',
  svm:            'SVM',
  xgboost:        'XGBoost',
  neural_network: 'Neural Network',
  QSVC:           'QSVC',
  VQC:            'VQC',
}

export const isQuantum = (name) => ['QSVC', 'VQC'].includes(name)
