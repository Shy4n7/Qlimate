import React, { useEffect, useRef, useState } from 'react'
import { geoMercator, geoPath } from 'd3-geo'

export default function IndiaMap({ selectedState, onSelect }) {
  const containerRef = useRef(null)
  const [geojson, setGeojson]     = useState(null)
  const [paths, setPaths]         = useState([])
  const [hoveredState, setHovered] = useState(null)
  const [tooltip, setTooltip]     = useState({ visible: false, x: 0, y: 0, name: '' })
  const [dims, setDims]           = useState({ width: 500, height: 540 })

  useEffect(() => {
    fetch('/india_states.geojson')
      .then((r) => r.json())
      .then(setGeojson)
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!geojson) return
    const { width, height } = dims

    const projection = geoMercator().fitSize([width, height], geojson)
    const pathGen    = geoPath().projection(projection)

    setPaths(
      geojson.features.map((f) => ({
        d:    pathGen(f),
        name: f.properties.NAME_1 || f.properties.name || '',
      }))
    )
  }, [geojson, dims])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const obs = new ResizeObserver(([entry]) => {
      const w = entry.contentRect.width
      setDims({ width: w, height: Math.round(w * 1.08) })
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const handleMouseMove = (e, name) => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    setTooltip({ visible: true, x: e.clientX - rect.left + 12, y: e.clientY - rect.top - 8, name })
  }

  return (
    <div ref={containerRef} className="relative w-full select-none">
      <svg
        width={dims.width}
        height={dims.height}
        className="w-full"
        style={{ display: 'block' }}
      >
        {paths.map(({ d, name }) => {
          const isSelected = name === selectedState
          const isHovered  = name === hoveredState

          return (
            <path
              key={name}
              d={d}
              fill={
                isSelected ? '#7c3aed'
                : isHovered  ? '#334155'
                : '#1e293b'
              }
              stroke={isSelected ? '#a78bfa' : '#475569'}
              strokeWidth={isSelected ? 1.5 : 0.6}
              style={{ cursor: 'pointer', transition: 'fill 0.15s' }}
              onClick={() => onSelect(name)}
              onMouseEnter={(e) => { setHovered(name); handleMouseMove(e, name) }}
              onMouseMove={(e)  => handleMouseMove(e, name)}
              onMouseLeave={() => { setHovered(null); setTooltip((t) => ({ ...t, visible: false })) }}
            />
          )
        })}
      </svg>

      {tooltip.visible && (
        <div
          className="absolute pointer-events-none bg-slate-800 border border-slate-600 text-slate-200 text-xs px-2 py-1 rounded shadow-lg whitespace-nowrap z-10"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.name}
          {tooltip.name === selectedState && (
            <span className="ml-1 text-violet-400">✓</span>
          )}
        </div>
      )}

      {!geojson && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
          Loading map…
        </div>
      )}
    </div>
  )
}
