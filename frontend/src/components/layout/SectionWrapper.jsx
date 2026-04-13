import React from 'react'

export default function SectionWrapper({ id, title, subtitle, children, dark = false }) {
  return (
    <section
      id={id}
      className={`py-20 px-4 section-divider ${dark ? 'bg-slate-900' : 'bg-slate-950'}`}
    >
      <div className="max-w-6xl mx-auto">
        {title && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-white mb-2">{title}</h2>
            {subtitle && <p className="text-slate-400 text-sm max-w-2xl">{subtitle}</p>}
          </div>
        )}
        {children}
      </div>
    </section>
  )
}
