import React from 'react'
import { MONTHS } from '../api/predict'

const HISTORICAL_YEARS = Array.from({ length: 2024 - 1995 + 1 }, (_, i) => 1995 + i)
const FUTURE_YEARS     = Array.from({ length: 2030 - 2025 + 1 }, (_, i) => 2025 + i)

export default function PredictionControls({
  selectedState,
  selectedMonth,
  selectedYear,
  onMonthChange,
  onYearChange,
  onPredict,
  isLoading,
}) {
  const ready = selectedState && selectedMonth && selectedYear && !isLoading
  const isFuture = selectedYear && selectedYear > 2024

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-end">
        <div className="flex-1">
          <label className="block text-slate-400 text-xs mb-1">Selected state</label>
          <div className={`px-3 py-2 rounded-lg border text-sm font-medium ${
            selectedState
              ? 'border-violet-700 bg-violet-950/40 text-violet-200'
              : 'border-slate-700 bg-slate-800/40 text-slate-500'
          }`}>
            {selectedState || 'Click a state on the map'}
          </div>
        </div>

        <div>
          <label className="block text-slate-400 text-xs mb-1">Month</label>
          <select
            value={selectedMonth || ''}
            onChange={(e) => onMonthChange(Number(e.target.value))}
            className="px-3 py-2 rounded-lg border border-slate-700 bg-slate-800 text-slate-200 text-sm focus:border-violet-600 focus:outline-none"
          >
            <option value="">Month</option>
            {MONTHS.map((m, i) => (
              <option key={m} value={i + 1}>{m}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-slate-400 text-xs mb-1">
            Year
            {isFuture && (
              <span className="ml-2 text-amber-400 font-semibold">Projection</span>
            )}
          </label>
          <select
            value={selectedYear || ''}
            onChange={(e) => onYearChange(Number(e.target.value))}
            className={`px-3 py-2 rounded-lg border bg-slate-800 text-sm focus:outline-none ${
              isFuture
                ? 'border-amber-600 text-amber-300 focus:border-amber-500'
                : 'border-slate-700 text-slate-200 focus:border-violet-600'
            }`}
          >
            <option value="">Year</option>
            <optgroup label="Historical (1995–2024)">
              {HISTORICAL_YEARS.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </optgroup>
            <optgroup label="Projection (2025–2030)">
              {FUTURE_YEARS.map((y) => (
                <option key={y} value={y}>{y} *</option>
              ))}
            </optgroup>
          </select>
        </div>

        <button
          onClick={onPredict}
          disabled={!ready}
          className={`px-5 py-2 rounded-lg text-sm font-semibold transition-colors ${
            ready
              ? 'bg-violet-700 hover:bg-violet-600 text-white cursor-pointer'
              : 'bg-slate-700 text-slate-500 cursor-not-allowed'
          }`}
        >
          {isLoading ? 'Predicting…' : 'Predict →'}
        </button>
      </div>

      {isFuture ? (
        <p className="text-amber-600 text-xs leading-relaxed border-t border-amber-900/40 pt-3">
          * Projection mode: no satellite data exists for {selectedYear}.
          Both models predict using the 30-year historical average for this state and month.
          This is a baseline estimate, not a forecast.
        </p>
      ) : (
        <p className="text-slate-600 text-xs leading-relaxed border-t border-slate-800 pt-3">
          Predictions are based on historical patterns learned from 1995–2024 satellite data,
          not physical climate simulation. Educational demonstration only — not a weather forecast.
        </p>
      )}
    </div>
  )
}
