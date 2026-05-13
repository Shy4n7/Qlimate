import Plot from 'react-plotly.js'
import { MONTHS } from '../api/predict'

/**
 * ForecastDivergenceChart
 *
 * Line chart of classical (blue, #3b82f6) vs quantum (violet, #8b5cf6)
 * temperature projections for years 2025-2035 using react-plotly.js.
 *
 * Props:
 *   forecastData  - { years, classical_temps, quantum_temps, is_mock } | null
 *   state         - Indian state name string
 *   month         - 1-based month integer
 *   isLoading     - boolean; show skeleton while true and forecastData is null
 */
export default function ForecastDivergenceChart({ forecastData, state, month, isLoading }) {
  // Show loading skeleton while data is being fetched
  if (forecastData === null && isLoading) {
    return (
      <div className="w-full rounded-xl bg-[#1e293b] p-4">
        <div className="animate-pulse space-y-3">
          <div className="h-4 w-48 rounded bg-slate-700" />
          <div className="h-64 w-full rounded bg-slate-700" />
        </div>
      </div>
    )
  }

  // Return nothing when there is no data and we are not loading
  if (forecastData === null) {
    return null
  }

  const { years, classical_temps, quantum_temps } = forecastData

  // MONTHS is 0-indexed; month prop is 1-based
  const monthName = MONTHS[(month - 1 + 12) % 12]

  const classicalTrace = {
    x: years,
    y: classical_temps,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Classical (XGBoost)',
    line: { color: '#3b82f6', width: 2 },
    marker: { color: '#3b82f6', size: 6 },
  }

  const quantumTrace = {
    x: years,
    y: quantum_temps,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Quantum (QSVR)',
    line: { color: '#8b5cf6', width: 2 },
    marker: { color: '#8b5cf6', size: 6 },
  }

  const layout = {
    title: {
      text: "2025\u20132035 Temperature Projection \u2014 " + state + ", " + monthName,
      font: { color: '#94a3b8', size: 14 },
    },
    paper_bgcolor: '#0f172a',
    plot_bgcolor: '#1e293b',
    font: { color: '#94a3b8' },
    xaxis: {
      title: { text: 'Year', font: { color: '#94a3b8' } },
      tickfont: { color: '#94a3b8' },
      gridcolor: '#334155',
      linecolor: '#475569',
      tickmode: 'array',
      tickvals: years,
    },
    yaxis: {
      title: { text: 'Temperature (\u00b0C)', font: { color: '#94a3b8' } },
      tickfont: { color: '#94a3b8' },
      gridcolor: '#334155',
      linecolor: '#475569',
    },
    legend: {
      font: { color: '#94a3b8' },
      bgcolor: 'rgba(0,0,0,0)',
    },
    margin: { t: 50, r: 20, b: 50, l: 60 },
    autosize: true,
  }

  const config = {
    displayModeBar: false,
    responsive: true,
  }

  return (
    <div className="w-full rounded-xl bg-[#0f172a] p-4">
      <Plot
        data={[classicalTrace, quantumTrace]}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '320px' }}
        useResizeHandler
      />
    </div>
  )
}
