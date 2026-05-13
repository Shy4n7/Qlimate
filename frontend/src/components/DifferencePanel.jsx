import React from 'react'

/**
 * Formats a temperature delta as "Δ +1.2 °C" or "Δ −0.8 °C".
 * Uses the proper minus sign (U+2212) instead of a hyphen.
 */
function formatDelta(delta) {
  if (delta == null) return 'Δ — °C'
  const sign = delta >= 0 ? '+' : '−'
  return `Δ ${sign}${Math.abs(delta).toFixed(1)} °C`
}

/**
 * Formats a predicted temperature value as "28.4 °C".
 */
function formatTemp(temp) {
  if (temp == null) return null
  return `${temp.toFixed(1)} °C`
}

export default function DifferencePanel({ result }) {
  if (!result) return null

  const { classical, quantum, temp_delta_c, divergence_note, is_mock } = result

  const classicalTemp = formatTemp(classical?.predicted_temp_c)
  const quantumTemp = formatTemp(quantum?.predicted_temp_c)
  const quantumUnavailable = quantum?.predicted_temp_c == null

  return (
    <div className="card border border-slate-700 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="text-slate-300 font-semibold text-sm">Model Comparison</div>
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border bg-slate-800/60 border-slate-600 text-slate-300 font-mono">
            {formatDelta(temp_delta_c)}
          </span>
          {is_mock && (
            <span className="text-slate-600 text-xs italic">mock data</span>
          )}
        </div>
      </div>

      {/* Side-by-side temperature cards */}
      <div className="grid grid-cols-2 gap-3 text-xs text-center">
        <div className="bg-blue-950/30 border border-blue-900 rounded-lg p-3">
          <div className="text-slate-400 mb-1">Classical predicted</div>
          <div className="text-blue-200 font-semibold text-base">
            {classicalTemp ?? '—'}
          </div>
          <div className="text-slate-500 mt-0.5">{classical?.model ?? 'Classical'}</div>
        </div>
        <div className="bg-violet-950/30 border border-violet-900 rounded-lg p-3">
          <div className="text-slate-400 mb-1">Quantum predicted</div>
          {quantumUnavailable ? (
            <div className="text-slate-500 font-semibold text-base italic">
              Quantum unavailable
            </div>
          ) : (
            <div className="text-violet-200 font-semibold text-base">
              {quantumTemp}
            </div>
          )}
          <div className="text-slate-500 mt-0.5">{quantum?.model ?? 'Quantum'}</div>
        </div>
      </div>

      {/* Divergence note */}
      {divergence_note && (
        <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
          <div className="text-slate-400 text-xs font-semibold mb-1">Divergence analysis</div>
          <p className="text-slate-300 text-xs leading-relaxed">{divergence_note}</p>
        </div>
      )}
    </div>
  )
}
