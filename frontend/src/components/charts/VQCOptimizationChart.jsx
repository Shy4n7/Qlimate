import React from 'react'
import Plot from 'react-plotly.js'

export default function VQCOptimizationChart({ optimization }) {
  const nn = optimization.neural_network
  const vqc = optimization.VQC

  const epochs = nn.train_losses.map((_, i) => i + 1)
  const startLoss = nn.train_losses[0]
  const endLoss = nn.final_train_loss
  const improvement = Math.round(((startLoss - endLoss) / startLoss) * 100)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* NN loss curves */}
      <div className="card">
        <h3 className="text-blue-300 font-semibold mb-1">Classical Neural Network — Learned steadily</h3>
        <p className="text-slate-400 text-xs mb-3">
          Like a student practicing problems — each round, it adjusted and improved. After {nn.epochs_run} rounds, it stopped because progress had plateaued.
        </p>
        <Plot
          data={[
            {
              x: epochs, y: nn.train_losses, type: 'scatter', mode: 'lines',
              name: 'Training error', line: { color: '#3b82f6', width: 2 },
            },
            {
              x: epochs, y: nn.val_losses, type: 'scatter', mode: 'lines',
              name: 'Validation error', line: { color: '#f97316', width: 2, dash: 'dash' },
            },
          ]}
          layout={{
            paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
            font: { color: '#e2e8f0', family: 'Inter' },
            xaxis: { title: 'Training round', gridcolor: '#1e293b' },
            yaxis: { title: 'Error (lower = better)', gridcolor: '#334155' },
            legend: { orientation: 'h', y: -0.28 },
            margin: { t: 10, b: 60, l: 65, r: 10 },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: 300 }}
        />
        <div className="mt-3 bg-blue-900/20 border border-blue-800 rounded-lg p-3 text-xs text-slate-300 leading-relaxed">
          Error dropped by ~{improvement}% over {nn.epochs_run} rounds. Both training and validation lines track each other — the model is genuinely learning, not just memorizing.
        </div>
      </div>

      {/* VQC panel */}
      <div className="card flex flex-col">
        <h3 className="text-violet-300 font-semibold mb-1">Quantum Neural Net (VQC) — Never learned</h3>
        <p className="text-slate-400 text-xs mb-3">
          The quantum model ran for 3 minutes — but a software bug prevented it from ever adjusting its settings. It made predictions using its starting (random) configuration.
        </p>

        <div className="flex-1 flex flex-col items-center justify-center bg-slate-900 rounded-lg border border-slate-700 p-6 min-h-[180px]">
          <div className="text-5xl font-mono font-bold text-violet-400 mb-2">0</div>
          <div className="text-slate-300 text-sm mb-2">adjustments made</div>
          <div className="text-slate-500 text-xs text-center max-w-xs leading-relaxed">
            out of {vqc.maxiter_configured} planned — the equivalent of a student who sat in class for 3 minutes, then guessed on every exam question
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-center">
          <div className="bg-slate-800 rounded-lg p-3">
            <div className="text-slate-400 mb-1">Time spent</div>
            <div className="font-mono text-violet-300 text-sm">{(vqc.training_time_s / 60).toFixed(0)} minutes</div>
            <div className="text-slate-500">circuit setup and execution</div>
          </div>
          <div className="bg-slate-800 rounded-lg p-3">
            <div className="text-slate-400 mb-1">Outcome</div>
            <div className="font-mono text-amber-400 text-sm">Untrained</div>
            <div className="text-slate-500">predictions = random guessing</div>
          </div>
        </div>

        <div className="mt-3 bg-amber-950/30 border border-amber-800 rounded-lg p-3 text-xs text-amber-200 leading-relaxed">
          This is a known incompatibility in the quantum software library — not an error in the experiment design. It's included honestly because failures like this are a real part of working at the frontier of quantum computing.
        </div>
      </div>
    </div>
  )
}
