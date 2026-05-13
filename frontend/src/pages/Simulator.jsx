import React, { useState, useEffect } from 'react'
import IndiaMap from '../components/IndiaMap'
import PredictionControls from '../components/PredictionControls'
import TemperatureOutputPanel from '../components/TemperatureOutputPanel'
import ForecastDivergenceChart from '../components/ForecastDivergenceChart'
import DifferencePanel from '../components/DifferencePanel'
import { predict, fetchForecast } from '../api/predict'

export default function Simulator() {
  const [selectedState, setSelectedState] = useState(null)
  const [selectedMonth, setSelectedMonth] = useState(null)
  const [selectedYear, setSelectedYear]   = useState(null)
  const [result, setResult]               = useState(null)
  const [forecastData, setForecastData]   = useState(null)
  const [isLoading, setIsLoading]         = useState(false)

  useEffect(() => {
    if (!selectedState || !selectedMonth) {
      setForecastData(null)
      return
    }
    fetchForecast(selectedState, selectedMonth).then(setForecastData)
  }, [selectedState, selectedMonth])

  const handlePredict = async () => {
    if (!selectedState || !selectedMonth || !selectedYear) return
    setIsLoading(true)
    setResult(null)
    try {
      const data = await predict(selectedState, selectedMonth, selectedYear)
      setResult(data)
    } finally {
      setIsLoading(false)
    }
  }

  const handleStateSelect = (name) => {
    setSelectedState(name)
    setResult(null)
  }

  return (
    <div className="min-h-screen bg-slate-950 pt-16">
      <div className="max-w-7xl mx-auto px-4 py-10">

        {/* Header */}
        <div className="mb-8">
          <div className="text-slate-500 text-xs font-mono mb-2 tracking-widest uppercase">
            Qlimate - Interactive Simulator
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">
            Temperature Prediction Simulator
          </h1>
          <p className="text-slate-400 text-sm max-w-2xl">
            Click a state on the map, select a month, and see how{' '}
            <span className="classical-text font-medium">classical AI</span> and{' '}
            <span className="quantum-text font-medium">quantum AI</span> predict
            surface temperature from the same historical climate data.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

          {/* Left - Map */}
          <div className="lg:col-span-2">
            <div className="card sticky top-20">
              <div className="text-slate-400 text-xs mb-3 font-medium">
                Click a state to select it
              </div>
              <IndiaMap
                selectedState={selectedState}
                onSelect={handleStateSelect}
              />
            </div>
          </div>

          {/* Right - Controls + Results */}
          <div className="lg:col-span-3 space-y-5">

            <div className="card">
              <PredictionControls
                selectedState={selectedState}
                selectedMonth={selectedMonth}
                selectedYear={selectedYear}
                onMonthChange={setSelectedMonth}
                onYearChange={setSelectedYear}
                onPredict={handlePredict}
                isLoading={isLoading}
              />
            </div>

            <TemperatureOutputPanel
              result={result}
              state={selectedState}
              month={selectedMonth}
              year={selectedYear}
              isLoading={isLoading}
            />

            {(result || isLoading) && (
              <DifferencePanel result={result} />
            )}

            {selectedState && selectedMonth && (
              <ForecastDivergenceChart
                forecastData={forecastData}
                state={selectedState}
                month={selectedMonth}
              />
            )}

            {result && result.is_future && (
              <div className="card border bg-amber-950/20 border-amber-800">
                <div className="text-amber-400 text-xs font-semibold mb-2 uppercase tracking-widest">
                  Temperature Forecast - {result.year}
                </div>
                <p className="text-slate-300 text-sm">
                  No satellite data exists for <span className="text-white font-medium">{result.year}</span>.
                  Both models produced a temperature forecast using the 30-year historical average for{' '}
                  <span className="text-white font-medium">{result.state}</span> in this month.
                </p>
                <p className="text-amber-700 text-xs mt-2">
                  This is a baseline temperature forecast - not a climate simulation. Real future conditions
                  may differ due to climate change trends not captured in the training data.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}