import React from 'react'

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/90 backdrop-blur border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <div className="font-mono text-sm font-semibold text-slate-300">
          <span className="classical-text">Classical</span>
          <span className="text-slate-500 mx-1">vs</span>
          <span className="quantum-text">Quantum</span>
          <span className="text-slate-500 ml-2 font-normal">— Qlimate</span>
        </div>
        <div className="text-slate-500 text-xs">
          Temperature Prediction Simulator · India 1995–2035
        </div>
      </div>
    </nav>
  )
}
