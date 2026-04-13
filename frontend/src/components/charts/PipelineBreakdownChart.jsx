import React from 'react'
import Plot from 'react-plotly.js'
import { MODEL_LABELS, MODEL_COLORS } from '../../utils/formatters'

const PLAIN_LABELS = {
  random_forest: 'Random Forest',
  svm: 'SVM',
  xgboost: 'XGBoost',
  neural_network: 'Neural Net',
  QSVC: 'Quantum SVM',
  VQC: 'Quantum Neural Net',
}

export default function PipelineBreakdownChart({ efficiency }) {
  const models = Object.keys(efficiency.models)
  const labels = models.map((m) => PLAIN_LABELS[m] || m)

  const trainTimes = models.map((m) => {
    const t = efficiency.models[m].training_time_s
    return t != null ? t / 60 : 0.5
  })
  const predTimes = models.map((m) => {
    const t = efficiency.models[m].prediction_time_s
    return t != null ? t / 60 : 0
  })
  const colors = models.map((m) => MODEL_COLORS[m] || '#6b7280')
  const trainLabels = models.map((m) => {
    const t = efficiency.models[m].training_time_s
    return t != null ? `${(t / 60).toFixed(1)}m` : '< 1 min'
  })

  return (
    <div className="space-y-4">
      <Plot
        data={[
          {
            type: 'bar',
            name: 'Training time',
            x: labels,
            y: trainTimes,
            marker: { color: colors, opacity: 0.9 },
            text: trainLabels,
            textposition: 'outside',
            textfont: { size: 11, color: '#e2e8f0' },
          },
          {
            type: 'bar',
            name: 'Prediction time (on test data)',
            x: labels,
            y: predTimes,
            marker: { color: colors, opacity: 0.4 },
            text: predTimes.map((t) => (t > 0.1 ? `${t.toFixed(0)}m` : '')),
            textposition: 'outside',
            textfont: { size: 11, color: '#94a3b8' },
          },
        ]}
        layout={{
          barmode: 'stack',
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { color: '#e2e8f0', family: 'Inter' },
          xaxis: { gridcolor: '#1e293b' },
          yaxis: {
            title: 'Time (minutes) — log scale',
            gridcolor: '#334155',
            type: 'log',
          },
          legend: { orientation: 'h', y: -0.22 },
          margin: { t: 20, b: 70, l: 75, r: 20 },
          shapes: [
            {
              type: 'line', x0: 3.5, x1: 3.5, y0: 0, y1: 1, yref: 'paper',
              line: { color: '#475569', dash: 'dot', width: 1.5 },
            },
          ],
          annotations: [
            { x: 1.5, y: 1.06, yref: 'paper', text: '← Classical', showarrow: false, font: { color: '#60a5fa', size: 11 } },
            { x: 4.5, y: 1.06, yref: 'paper', text: 'Quantum →', showarrow: false, font: { color: '#a78bfa', size: 11 } },
          ],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: 360 }}
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs text-center">
        <div className="card">
          <div className="text-slate-400 mb-1">Classical training</div>
          <div className="text-blue-300 font-semibold text-sm">Under 1 minute</div>
          <div className="text-slate-500">fast enough to try many configurations</div>
        </div>
        <div className="card border border-violet-800">
          <div className="text-slate-400 mb-1">Quantum Kernel (QSVC)</div>
          <div className="text-violet-300 font-semibold text-sm">46 min training + 35 min predicting</div>
          <div className="text-slate-500">on 1/17th of the data</div>
        </div>
        <div className="card">
          <div className="text-slate-400 mb-1">Note on the chart</div>
          <div className="text-amber-300 font-semibold text-sm">Log scale</div>
          <div className="text-slate-500">each step up = 10× more time — bars are compressed to fit</div>
        </div>
      </div>

      <p className="text-slate-600 text-xs text-center">
        Classical training times were under 1 minute and were not individually measured. Quantum times are from actual recorded runs.
      </p>
    </div>
  )
}
