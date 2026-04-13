import React from 'react'
import Plot from 'react-plotly.js'

export default function KernelMatrixChart({ kernelStats }) {
  const sim = kernelStats.simulator
  const hw = kernelStats.hardware_ibm_fez

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Simulator heatmap */}
        <div className="card">
          <h3 className="text-blue-300 font-semibold mb-1">Simulated on a regular computer</h3>
          <p className="text-slate-400 text-xs mb-3">
            50 months compared against 50 others. Bright = very similar weather. Dark = very different.
          </p>
          <Plot
            data={[{
              z: sim.matrix_data,
              type: 'heatmap',
              colorscale: 'Magma',
              zmin: 0, zmax: 1,
              showscale: true,
              colorbar: {
                title: 'Similarity',
                titlefont: { color: '#94a3b8' },
                tickfont: { color: '#94a3b8' },
                tickvals: [0, 0.5, 1],
                ticktext: ['Different', 'Neutral', 'Identical'],
              },
            }]}
            layout={{
              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
              font: { color: '#e2e8f0' },
              xaxis: { title: 'Month index', gridcolor: '#1e293b', tickfont: { size: 10 } },
              yaxis: { title: 'Month index', gridcolor: '#1e293b', tickfont: { size: 10 } },
              margin: { t: 10, b: 55, l: 55, r: 10 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: 300 }}
          />
          <div className="mt-3 bg-slate-800 rounded-lg p-3 text-xs text-slate-300 leading-relaxed">
            The bright diagonal is every month compared to itself — always 100% similar. The varied pattern off-diagonal means the model can tell different climate types apart clearly.
          </div>
        </div>

        {/* Hardware panel */}
        <div className="card flex flex-col gap-4">
          <div>
            <h3 className="text-violet-300 font-semibold mb-1">
              On a real IBM Quantum chip
            </h3>
            <p className="text-slate-400 text-xs mb-4">
              127-qubit superconducting processor in New York
            </p>

            <div className="mb-3">
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Rows completed before quota ran out</span>
                <span className="font-mono text-white">
                  {hw.kernel_rows_evaluated} of {hw.kernel_rows_planned}
                </span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-2">
                <div
                  className="bg-violet-500 h-2 rounded-full"
                  style={{ width: `${(hw.kernel_rows_evaluated / hw.kernel_rows_planned) * 100}%` }}
                />
              </div>
              <p className="text-amber-400 text-xs mt-1.5">
                IBM gives free users 10 minutes per month. Two rows used it all up — each row took ~8 minutes of quantum compute time.
              </p>
            </div>
          </div>

          <div className="bg-slate-900 rounded-lg p-4 border border-slate-700 flex-1">
            <div className="text-slate-300 text-xs font-semibold mb-2">What noise does to the results</div>
            <p className="text-slate-400 text-xs leading-relaxed">
              Real quantum chips are imperfect. Random errors creep into every calculation — like static on an old radio. These errors push all similarity scores toward 50%, blurring the difference between "very similar" and "very different" months. The result: the model loses confidence in its classifications.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs text-center">
            <div className="bg-blue-900/30 border border-blue-800 rounded-lg p-3">
              <div className="text-slate-400 mb-1">Simulator range</div>
              <div className="text-blue-300 font-semibold text-sm">0% – 92%</div>
              <div className="text-slate-500">high contrast = clear decisions</div>
            </div>
            <div className="bg-violet-900/30 border border-violet-800 rounded-lg p-3">
              <div className="text-slate-400 mb-1">Hardware (expected)</div>
              <div className="text-violet-300 font-semibold text-sm">~30% – 70%</div>
              <div className="text-slate-500">noise squeezes the range inward</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
