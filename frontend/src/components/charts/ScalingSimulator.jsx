import React, { useState, useMemo } from 'react'
import Plot from 'react-plotly.js'

const QSVC_TIME_AT_400 = 274.6
const VQC_TIME_AT_400 = 198.0
const N_ANCHOR = 400

function qsvcTime(n) {
  return (QSVC_TIME_AT_400 * (n / N_ANCHOR) ** 2) / 60
}

function vqcTime(n, maxiter = 150) {
  const perSamplePerIter = VQC_TIME_AT_400 / (N_ANCHOR * maxiter)
  return (perSamplePerIter * n * maxiter) / 60
}

function rfTime(n) {
  return (0.008 * n * Math.log2(Math.max(n, 2))) / Math.log2(400) / 60
}

function consequence(n) {
  if (n <= 400) {
    const t = qsvcTime(n)
    if (t < 1) return `Quantum takes under a minute — manageable, but already slower than classical.`
    return `Quantum takes ${t.toFixed(0)} minutes for ${n} examples. Classical is done in seconds.`
  }
  if (n <= 1000) {
    const t = qsvcTime(n)
    return `At ${n} examples, quantum takes ${t.toFixed(0)} minutes. Classical still under a minute. The gap is widening fast.`
  }
  if (n <= 3000) {
    const hrs = qsvcTime(n) / 60
    return `At ${n} examples, quantum would need ${hrs.toFixed(1)} hours. Classical: still under a minute. This is why quantum was capped at 400.`
  }
  const hrs = qsvcTime(n) / 60
  return `At ${n} examples, quantum would need ~${Math.round(hrs)} hours — impractical. This is the wall quantum computing hits today.`
}

export default function ScalingSimulator() {
  const [nSamples, setNSamples] = useState(400)

  const n_range = useMemo(() => {
    const pts = []
    for (let n = 50; n <= 7000; n += 50) pts.push(n)
    return pts
  }, [])

  const qsvcCurve = n_range.map(qsvcTime)
  const vqcCurve = n_range.map(vqcTime)
  const rfCurve = n_range.map(rfTime)

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <label className="text-slate-300 text-sm font-medium min-w-max">
          Training examples:
        </label>
        <input
          type="range"
          min={50}
          max={7000}
          step={50}
          value={nSamples}
          onChange={(e) => setNSamples(Number(e.target.value))}
          className="flex-1 accent-violet-500"
        />
        <span className="font-mono text-violet-300 min-w-[70px] text-right text-sm">
          {nSamples.toLocaleString()}
        </span>
      </div>

      <div className="card bg-amber-950/30 border border-amber-800">
        <p className="text-amber-200 text-sm leading-relaxed">{consequence(nSamples)}</p>
      </div>

      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="card py-3">
          <div className="text-blue-400 font-semibold text-xs mb-1">Classical (Random Forest)</div>
          <div className="text-white font-mono text-xl">{rfTime(nSamples) < 0.1 ? '< 0.1 min' : `${rfTime(nSamples).toFixed(1)} min`}</div>
          <div className="text-slate-500 text-xs mt-1">Gets a little slower with more data</div>
        </div>
        <div className="card py-3 border border-violet-800">
          <div className="text-violet-400 font-semibold text-xs mb-1">Quantum Kernel (QSVC)</div>
          <div className="text-white font-mono text-xl">
            {qsvcTime(nSamples) >= 60
              ? `~${(qsvcTime(nSamples) / 60).toFixed(0)} hrs`
              : `${qsvcTime(nSamples).toFixed(0)} min`}
          </div>
          <div className="text-slate-500 text-xs mt-1">Doubles when data doubles — then doubles again</div>
        </div>
        <div className="card py-3">
          <div className="text-purple-400 font-semibold text-xs mb-1">Quantum Neural Net (VQC)</div>
          <div className="text-white font-mono text-xl">{vqcTime(nSamples).toFixed(0)} min</div>
          <div className="text-slate-500 text-xs mt-1">Scales linearly — but still much slower</div>
        </div>
      </div>

      <Plot
        data={[
          {
            x: n_range, y: rfCurve, type: 'scatter', mode: 'lines',
            name: 'Classical (Random Forest)', line: { color: '#3b82f6', width: 2 },
          },
          {
            x: n_range, y: qsvcCurve, type: 'scatter', mode: 'lines',
            name: 'Quantum Kernel (QSVC)', line: { color: '#7c3aed', width: 2.5 },
          },
          {
            x: n_range, y: vqcCurve, type: 'scatter', mode: 'lines',
            name: 'Quantum Neural Net (VQC)', line: { color: '#a855f7', width: 2, dash: 'dash' },
          },
          {
            x: [nSamples], y: [rfTime(nSamples)], type: 'scatter', mode: 'markers',
            showlegend: false, marker: { color: '#3b82f6', size: 11, symbol: 'circle' },
          },
          {
            x: [nSamples], y: [qsvcTime(nSamples)], type: 'scatter', mode: 'markers',
            showlegend: false, marker: { color: '#7c3aed', size: 11, symbol: 'circle' },
          },
          {
            x: [nSamples], y: [vqcTime(nSamples)], type: 'scatter', mode: 'markers',
            showlegend: false, marker: { color: '#a855f7', size: 11, symbol: 'circle' },
          },
          {
            x: [400], y: [QSVC_TIME_AT_400 / 60], type: 'scatter', mode: 'markers',
            name: '★ Actual measured time (QSVC at 400)', marker: { color: '#f59e0b', size: 14, symbol: 'star' },
          },
        ]}
        layout={{
          paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
          font: { color: '#e2e8f0', family: 'Inter' },
          xaxis: { title: 'Number of training examples', gridcolor: '#1e293b' },
          yaxis: { title: 'Time to train (minutes)', gridcolor: '#334155' },
          legend: { orientation: 'h', y: -0.28, font: { size: 11 } },
          margin: { t: 10, b: 70, l: 70, r: 20 },
          shapes: [
            {
              type: 'line', x0: 400, x1: 400, y0: 0, y1: 1,
              yref: 'paper', line: { color: '#f59e0b', dash: 'dot', width: 1 },
            },
          ],
          annotations: [
            {
              x: 440, y: 0.92, yref: 'paper',
              text: 'Quantum was<br>capped here',
              showarrow: false, font: { color: '#fbbf24', size: 10 }, xanchor: 'left',
            },
          ],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: 340 }}
      />
    </div>
  )
}
