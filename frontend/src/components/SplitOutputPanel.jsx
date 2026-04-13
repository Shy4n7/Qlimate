import React from 'react'
import { MONTHS } from '../api/predict'

const CLASS_COLORS = {
  Normal:       { bg: 'bg-slate-700',  border: 'border-slate-500',  text: 'text-slate-200' },
  Drought:      { bg: 'bg-amber-900/60', border: 'border-amber-600', text: 'text-amber-200' },
  Wet_Flood:    { bg: 'bg-blue-900/60',  border: 'border-blue-600',  text: 'text-blue-200'  },
  Heat_Extreme: { bg: 'bg-red-900/60',   border: 'border-red-600',   text: 'text-red-200'   },
  Cold_Extreme: { bg: 'bg-cyan-900/60',  border: 'border-cyan-600',  text: 'text-cyan-200'  },
}

function ConditionBadge({ label, displayLabel }) {
  const c = CLASS_COLORS[label] || CLASS_COLORS.Normal
  return (
    <span className={`inline-block px-3 py-1 rounded-full text-sm font-semibold border ${c.bg} ${c.border} ${c.text}`}>
      {displayLabel || label}
    </span>
  )
}

function ConfidenceBar({ value, color = 'violet' }) {
  const pct = Math.round(value * 100)
  const bar = {
    blue:   'bg-blue-500',
    violet: 'bg-violet-500',
    amber:  'bg-amber-500',
  }[color] || 'bg-violet-500'

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-400">
        <span>Confidence</span>
        <span className="font-mono text-white">{pct}%</span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2">
        <div
          className={`${bar} h-2 rounded-full transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function ModelCard({ side, data, state, month, year }) {
  const isClassical = side === 'classical'
  const border = isClassical ? 'border-blue-800' : 'border-violet-800'
  const heading = isClassical ? 'text-blue-300' : 'text-violet-300'
  const barColor = isClassical ? 'blue' : 'violet'
  const monthName = month ? MONTHS[month - 1] : ''

  return (
    <div className={`card border ${border} flex flex-col gap-4`}>
      <div>
        <div className={`font-bold text-base mb-0.5 ${heading}`}>
          {isClassical ? 'Classical AI' : 'Quantum AI'}
        </div>
        <div className="text-slate-500 text-xs">
          {isClassical ? `XGBoost · regular computer` : `QSVC · IBM quantum hardware`}
        </div>
      </div>

      {data ? (
        <>
          <div>
            <div className="text-slate-400 text-xs mb-2">
              Predicted condition for {state}{monthName ? `, ${monthName}` : ''}{year ? ` ${year}` : ''}:
            </div>
            <ConditionBadge label={data.label} displayLabel={data.label_display} />
          </div>

          <ConfidenceBar value={data.confidence} color={barColor} />

          {data.confidence_note && (
            <p className="text-slate-600 text-xs italic">{data.confidence_note}</p>
          )}

          <div className={`rounded-lg p-3 text-xs leading-relaxed space-y-1 ${isClassical ? 'bg-blue-950/30 border border-blue-900' : 'bg-violet-950/30 border border-violet-900'}`}>
            <div className="flex justify-between">
              <span className="text-slate-400">Training examples</span>
              <span className="text-white font-mono">{data.n_train.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Features used</span>
              <span className="text-white font-mono">{data.n_features}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Data compression</span>
              <span className={isClassical ? 'text-blue-300' : 'text-violet-300'}>
                {isClassical ? 'None' : '9 → 4 (PCA)'}
              </span>
            </div>
          </div>
        </>
      ) : (
        <div className="space-y-3 animate-pulse">
          <div className="h-4 bg-slate-700 rounded w-3/4" />
          <div className="h-8 bg-slate-700 rounded w-1/2" />
          <div className="h-2 bg-slate-700 rounded" />
          <div className="h-16 bg-slate-700 rounded" />
        </div>
      )}
    </div>
  )
}

export default function SplitOutputPanel({ result, state, month, year, isLoading }) {
  const classical = result?.classical ?? (isLoading ? null : undefined)
  const quantum   = result?.quantum   ?? (isLoading ? null : undefined)

  if (!result && !isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {['Classical AI', 'Quantum AI'].map((label) => (
          <div key={label} className="card border border-slate-800 flex items-center justify-center min-h-[200px]">
            <p className="text-slate-600 text-sm text-center">
              {label}<br />
              <span className="text-xs">Select a state and month to see predictions</span>
            </p>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <ModelCard side="classical" data={classical} state={state} month={month} year={year} />
      <ModelCard side="quantum"   data={quantum}   state={state} month={month} year={year} />
    </div>
  )
}
