import React from 'react'

const RAW_FEATURES = [
  { key: 'T2M', label: 'Temperature', icon: '🌡' },
  { key: 'QV2M', label: 'Humidity', icon: '💧' },
  { key: 'U10M', label: 'Wind (East)', icon: '💨' },
  { key: 'V10M', label: 'Wind (North)', icon: '💨' },
  { key: 'PS', label: 'Surface Pressure', icon: '⬇' },
  { key: 'SLP', label: 'Sea Level Pressure', icon: '🌊' },
  { key: 'PRECTOT', label: 'Precipitation', icon: '🌧' },
  { key: 'EVAP', label: 'Evaporation', icon: '☀' },
  { key: 'CLDTOT', label: 'Cloud Cover', icon: '☁' },
]

const PC_DESCRIPTIONS = [
  'Mostly temperature + pressure (biggest pattern)',
  'Mostly precipitation + humidity',
  'Wind patterns',
  'Remaining subtle variation',
]

export default function FeatureFlowDiagram({ dataEfficiency }) {
  const q = dataEfficiency.quantum
  const variances = q.pca_variance_per_component

  return (
    <div className="flex flex-col lg:flex-row items-start gap-4 lg:gap-2 py-4">
      {/* Raw features */}
      <div className="flex flex-col items-center min-w-[190px]">
        <div className="text-blue-400 font-semibold text-sm mb-3 text-center">
          9 climate measurements<br />
          <span className="text-slate-400 font-normal text-xs">collected by NASA satellite</span>
        </div>
        <div className="space-y-1 w-full">
          {RAW_FEATURES.map((f) => (
            <div key={f.key} className="bg-blue-900/40 border border-blue-800 rounded px-2 py-1.5 text-blue-200 text-xs flex items-center gap-2">
              <span>{f.icon}</span>
              <span>{f.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Arrow */}
      <div className="flex flex-col items-center gap-2 px-3 pt-16">
        <div className="text-slate-500 text-xs text-center leading-tight">compress &<br/>find patterns</div>
        <div className="text-slate-400 text-2xl">→</div>
        <div className="bg-slate-700 border border-slate-500 rounded px-2 py-1 text-slate-300 text-xs text-center">
          94% of info<br/>kept
        </div>
      </div>

      {/* PCA components */}
      <div className="flex flex-col items-center min-w-[200px]">
        <div className="text-violet-400 font-semibold text-sm mb-3 text-center">
          4 compressed patterns<br />
          <span className="text-slate-400 font-normal text-xs">ranked by how much they explain</span>
        </div>
        <div className="space-y-2 w-full">
          {variances.map((v, i) => (
            <div key={i} className="bg-violet-900/40 border border-violet-800 rounded px-3 py-2">
              <div className="flex justify-between items-center mb-1">
                <div className="text-violet-200 text-xs font-medium">Pattern {i + 1}</div>
                <div className="text-violet-300 text-xs font-mono">{(v * 100).toFixed(0)}%</div>
              </div>
              <div className="w-full bg-violet-900 rounded-full h-1.5">
                <div
                  className="bg-violet-400 h-1.5 rounded-full"
                  style={{ width: `${(v / variances[0]) * 100}%` }}
                />
              </div>
              <div className="text-slate-500 text-xs mt-1 leading-snug">{PC_DESCRIPTIONS[i]}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Arrow */}
      <div className="flex flex-col items-center gap-2 px-3 pt-16">
        <div className="text-slate-500 text-xs text-center leading-tight">encode as<br/>qubit angles</div>
        <div className="text-slate-400 text-2xl">→</div>
        <div className="bg-slate-700 border border-slate-500 rounded px-2 py-1 text-slate-300 text-xs text-center">
          one value<br/>= one dial
        </div>
      </div>

      {/* Qubits */}
      <div className="flex flex-col items-center min-w-[160px]">
        <div className="text-teal-400 font-semibold text-sm mb-3 text-center">
          4 qubits on a<br />quantum chip
          <br /><span className="text-slate-400 font-normal text-xs">IBM ibm_fez · New York</span>
        </div>
        <div className="space-y-3 w-full">
          {Array.from({ length: q.features_after_pca }, (_, i) => (
            <div key={i} className="bg-teal-900/40 border border-teal-700 rounded px-3 py-3 text-center">
              <div className="text-teal-300 text-base font-mono">|ψ{i}⟩</div>
              <div className="text-teal-500 text-xs mt-0.5">Qubit {i + 1} — rotated by Pattern {i + 1}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
