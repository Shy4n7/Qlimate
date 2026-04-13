import React from 'react'

export default function DifferencePanel({ result }) {
  if (!result) return null

  const { agreement, confidence_gap, difference_reason, classical, quantum, is_mock } = result
  const gapPct = Math.round(confidence_gap * 100)

  return (
    <div className="card border border-slate-700 space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="text-slate-300 font-semibold text-sm">Model Comparison</div>
        <div className="flex items-center gap-3">
          <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${
            agreement
              ? 'bg-green-900/40 border-green-700 text-green-300'
              : 'bg-amber-900/40 border-amber-700 text-amber-300'
          }`}>
            {agreement ? '✓ Agree' : '✗ Disagree'}
          </span>
          {is_mock && (
            <span className="text-slate-600 text-xs italic">mock data</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs text-center">
        <div className="bg-blue-950/30 border border-blue-900 rounded-lg p-3">
          <div className="text-slate-400 mb-1">Classical predicted</div>
          <div className="text-blue-200 font-semibold">{classical?.label_display || '—'}</div>
          <div className="text-slate-500 mt-0.5">{Math.round((classical?.confidence || 0) * 100)}% confident</div>
        </div>
        <div className="bg-violet-950/30 border border-violet-900 rounded-lg p-3">
          <div className="text-slate-400 mb-1">Quantum predicted</div>
          <div className="text-violet-200 font-semibold">{quantum?.label_display || '—'}</div>
          <div className="text-slate-500 mt-0.5">{Math.round((quantum?.confidence || 0) * 100)}% confident</div>
        </div>
      </div>

      <div>
        <div className="flex justify-between text-xs text-slate-400 mb-1">
          <span>Confidence gap</span>
          <span className="font-mono text-white">{gapPct}%</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-1.5">
          <div
            className="bg-amber-500 h-1.5 rounded-full transition-all duration-500"
            style={{ width: `${Math.min(gapPct * 2, 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-slate-600 mt-0.5">
          <span>Same confidence</span>
          <span>Large gap</span>
        </div>
      </div>

      <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
        <div className="text-slate-400 text-xs font-semibold mb-1">Why the difference?</div>
        <p className="text-slate-300 text-xs leading-relaxed">{difference_reason}</p>
      </div>
    </div>
  )
}
