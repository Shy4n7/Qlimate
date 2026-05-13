import React from 'react'
import { MONTHS } from '../api/predict'

function SkeletonCard({ side }) {
  const border = side === 'classical' ? 'border-blue-800' : 'border-violet-800'
  return (
    <div className={`card border ${border} flex flex-col gap-4`}>
      <div className="space-y-2 animate-pulse">
        <div className="h-4 bg-slate-700 rounded w-1/2" />
        <div className="h-3 bg-slate-700 rounded w-2/3" />
      </div>
      <div className="space-y-3 animate-pulse">
        <div className="h-10 bg-slate-700 rounded w-3/4" />
        <div className="h-3 bg-slate-700 rounded w-1/2" />
        <div className="h-16 bg-slate-700 rounded" />
      </div>
    </div>
  )
}

function ModelCard({ side, data, state, month, year }) {
  const isClassical = side === 'classical'
  const border  = isClassical ? 'border-blue-800'   : 'border-violet-800'
  const heading = isClassical ? 'text-blue-300'      : 'text-violet-300'
  const infoBg  = isClassical
    ? 'bg-blue-950/30 border border-blue-900'
    : 'bg-violet-950/30 border border-violet-900'
  const tempColor = isClassical ? 'text-blue-200' : 'text-violet-200'
  const monthName = month ? MONTHS[month - 1] : ''

  const isUnavailable = data && data.predicted_temp_c == null

  return (
    <div className={`card border ${border} flex flex-col gap-4`}>
      <div>
        <div className={`font-bold text-base mb-0.5 ${heading}`}>
          {isClassical ? 'Classical AI' : 'Quantum AI'}
        </div>
        <div className="text-slate-500 text-xs">
          {isClassical
            ? 'XGBoost Regressor · regular computer'
            : 'QSVR · quantum kernel machine'}
        </div>
      </div>

      {data ? (
        <>
          <div className="text-slate-400 text-xs">
            Predicted temperature for{' '}
            <span className="text-slate-200">{state}</span>
            {monthName ? <>, <span className="text-slate-200">{monthName}</span></> : null}
            {year ? <> <span className="text-slate-200">{year}</span></> : null}:
          </div>

          {isUnavailable ? (
            <div className="flex items-center gap-2">
              <span className="text-slate-500 text-sm italic">QSVR (unavailable)</span>
            </div>
          ) : (
            <div className={`text-4xl font-mono font-bold tracking-tight ${tempColor}`}>
              {data.predicted_temp_c.toFixed(1)}{' '}
              <span className="text-2xl font-normal text-slate-400">°C</span>
            </div>
          )}

          <div className={`rounded-lg p-3 text-xs leading-relaxed space-y-1 ${infoBg}`}>
            <div className="flex justify-between">
              <span className="text-slate-400">Model</span>
              <span className="text-white font-mono">
                {isUnavailable ? 'QSVR (unavailable)' : data.model}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Training examples</span>
              <span className="text-white font-mono">
                {data.n_train != null ? data.n_train.toLocaleString() : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Features used</span>
              <span className="text-white font-mono">
                {data.n_features != null ? data.n_features : '—'}
              </span>
            </div>
          </div>
        </>
      ) : (
        <div className="space-y-3 animate-pulse">
          <div className="h-3 bg-slate-700 rounded w-2/3" />
          <div className="h-10 bg-slate-700 rounded w-3/4" />
          <div className="h-16 bg-slate-700 rounded" />
        </div>
      )}
    </div>
  )
}

export default function TemperatureOutputPanel({ result, state, month, year, isLoading }) {
  if (isLoading && !result) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <SkeletonCard side="classical" />
        <SkeletonCard side="quantum" />
      </div>
    )
  }

  if (!result && !isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {['Classical AI', 'Quantum AI'].map((label) => (
          <div
            key={label}
            className="card border border-slate-800 flex items-center justify-center min-h-[200px]"
          >
            <p className="text-slate-600 text-sm text-center">
              {label}
              <br />
              <span className="text-xs">Select a state and month to see predictions</span>
            </p>
          </div>
        ))}
      </div>
    )
  }

  const classical = result?.classical ?? (isLoading ? null : undefined)
  const quantum   = result?.quantum   ?? (isLoading ? null : undefined)

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <ModelCard side="classical" data={classical} state={state} month={month} year={year} />
      <ModelCard side="quantum"   data={quantum}   state={state} month={month} year={year} />
    </div>
  )
}
