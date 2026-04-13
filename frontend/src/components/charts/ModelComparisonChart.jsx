import React from 'react'
import Plot from 'react-plotly.js'
import { MODEL_COLORS, MODEL_LABELS, isQuantum } from '../../utils/formatters'

const PLAIN_LABELS = {
  random_forest: 'Random Forest',
  svm: 'SVM',
  xgboost: 'XGBoost ★',
  neural_network: 'Neural Net',
  QSVC: 'Quantum SVM',
  VQC: 'Quantum VQC',
}

const EXPLANATIONS = {
  xgboost: 'Best classical — hundreds of decision trees voting together',
  neural_network: 'Layers of connected nodes, like a simplified brain',
  random_forest: 'Many decision trees, each trained on a random slice of data',
  svm: 'Finds the widest boundary between categories',
  QSVC: 'Quantum version of SVM — runs on IBM quantum hardware',
  VQC: 'Quantum neural net — never actually trained (software bug)',
}

export default function ModelComparisonChart({ performance }) {
  const models = Object.keys(performance.models)
  const labels = models.map((m) => PLAIN_LABELS[m] || m)
  const accuracy = models.map((m) => performance.models[m].accuracy)
  const colors = models.map((m) => MODEL_COLORS[m] || '#6b7280')
  const pcts = accuracy.map((a) => `${Math.round(a * 100)}%`)

  return (
    <div className="space-y-4">
      <Plot
        data={[{
          type: 'bar',
          x: labels,
          y: accuracy,
          marker: {
            color: colors,
            opacity: models.map((m) => isQuantum(m) ? 0.65 : 1.0),
            line: { color: colors, width: 1.5 },
          },
          text: pcts,
          textposition: 'outside',
          textfont: { size: 13, color: '#e2e8f0', family: 'Inter' },
        }]}
        layout={{
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#e2e8f0', family: 'Inter' },
          xaxis: { tickfont: { size: 12 }, gridcolor: '#1e293b' },
          yaxis: {
            range: [0, 0.85],
            gridcolor: '#334155',
            tickformat: ',.0%',
            title: 'Correct predictions',
          },
          margin: { t: 20, b: 60, l: 70, r: 20 },
          shapes: [
            {
              type: 'line', x0: 3.5, x1: 3.5, y0: 0, y1: 0.85,
              line: { color: '#475569', dash: 'dot', width: 1.5 },
            },
          ],
          annotations: [
            {
              x: 1.5, y: 0.78, text: '← Classical (regular computers)', showarrow: false,
              font: { color: '#60a5fa', size: 11 }, xref: 'x', yref: 'y',
            },
            {
              x: 4.5, y: 0.78, text: 'Quantum computers →', showarrow: false,
              font: { color: '#a78bfa', size: 11 }, xref: 'x', yref: 'y',
            },
          ],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: 360 }}
      />

      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
        {models.map((m) => (
          <div key={m} className={`rounded-lg px-3 py-2 text-xs border ${isQuantum(m) ? 'border-violet-800 bg-violet-950/30' : 'border-slate-700 bg-slate-800/40'}`}>
            <div className="font-semibold text-white mb-0.5">{PLAIN_LABELS[m]}</div>
            <div className="text-slate-400 leading-snug">{EXPLANATIONS[m]}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
