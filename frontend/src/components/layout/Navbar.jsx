import React, { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

const SECTIONS = [
  { id: 'landing',      label: 'Overview'    },
  { id: 'comparison',   label: 'Results'     },
  { id: 'scaling',      label: 'Speed'       },
  { id: 'features',     label: 'Data'        },
  { id: 'kernel',       label: 'Similarity'  },
  { id: 'optimization', label: 'Learning'    },
  { id: 'pipeline',     label: 'Time'        },
  { id: 'insights',     label: 'Takeaways'   },
]

export default function Navbar() {
  const [active, setActive]  = useState('landing')
  const location             = useLocation()
  const navigate             = useNavigate()
  const isSimulator          = location.pathname === '/simulator'

  useEffect(() => {
    if (isSimulator) return
    const observer = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) setActive(e.target.id) }),
      { threshold: 0.4 }
    )
    SECTIONS.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observer.observe(el)
    })
    return () => observer.disconnect()
  }, [isSimulator])

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/90 backdrop-blur border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 py-2 flex items-center justify-between">
        <button
          onClick={() => navigate('/')}
          className="font-mono text-sm font-semibold text-slate-300 hover:text-white transition-colors"
        >
          <span className="classical-text">Classical</span>
          <span className="text-slate-500 mx-1">vs</span>
          <span className="quantum-text">Quantum</span>
          <span className="text-slate-500 ml-2 font-normal">— Qlimate</span>
        </button>

        <div className="hidden md:flex items-center gap-1">
          {!isSimulator && SECTIONS.map(({ id, label }) => (
            <a
              key={id}
              href={`#${id}`}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                active === id
                  ? 'bg-violet-700 text-white'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {label}
            </a>
          ))}

          <button
            onClick={() => navigate(isSimulator ? '/' : '/simulator')}
            className={`ml-2 px-3 py-1 rounded text-xs font-medium transition-colors border ${
              isSimulator
                ? 'border-violet-600 text-violet-300 bg-violet-900/40 hover:bg-violet-900/60'
                : 'border-slate-600 text-slate-300 hover:bg-slate-800'
            }`}
          >
            {isSimulator ? '← Dashboard' : '⚡ Simulator'}
          </button>
        </div>
      </div>
    </nav>
  )
}
