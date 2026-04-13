import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/layout/Navbar'
import SectionWrapper from './components/layout/SectionWrapper'
import ModelComparisonChart from './components/charts/ModelComparisonChart'
import ScalingSimulator from './components/charts/ScalingSimulator'
import KernelMatrixChart from './components/charts/KernelMatrixChart'
import VQCOptimizationChart from './components/charts/VQCOptimizationChart'
import PipelineBreakdownChart from './components/charts/PipelineBreakdownChart'
import FeatureFlowDiagram from './components/charts/FeatureFlowDiagram'
import Simulator from './pages/Simulator'

import performance from './data/performance.json'
import efficiency from './data/efficiency.json'
import dataEfficiency from './data/data_efficiency.json'
import optimization from './data/optimization.json'
import kernelStats from './data/kernel_stats.json'
import practicality from './data/practicality.json'

import { fmtPct } from './utils/formatters'

function WhyCard({ title, children, accent = 'slate' }) {
  const border = { blue: 'border-blue-700', violet: 'border-violet-700', slate: 'border-slate-600', amber: 'border-amber-700' }
  const heading = { blue: 'text-blue-300', violet: 'text-violet-300', slate: 'text-slate-300', amber: 'text-amber-300' }
  return (
    <div className={`card border-l-4 ${border[accent]}`}>
      <div className={`font-semibold text-sm mb-2 ${heading[accent]}`}>{title}</div>
      <p className="text-slate-400 text-xs leading-relaxed">{children}</p>
    </div>
  )
}

function Dashboard() {
  const xgb = performance.models.xgboost
  const qsvc = performance.models.QSVC
  const bestClassicalPct = Math.round(xgb.accuracy * 100)
  const quantumPct = Math.round(qsvc.accuracy * 100)

  return (
    <div className="min-h-screen">
      {/* LANDING */}
      <section id="landing" className="min-h-screen flex flex-col items-center justify-center px-4 bg-gradient-to-b from-slate-900 to-slate-950 pt-20">
        <div className="max-w-3xl text-center">
          <div className="text-slate-500 text-sm font-mono mb-4 tracking-widest uppercase">
            Qlimate — A Research Experiment
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-5 leading-tight">
            Can a <span className="quantum-text">Quantum Computer</span> predict<br />
            Indian climate better than a <span className="classical-text">regular one</span>?
          </h1>
          <p className="text-slate-400 text-lg mb-4 leading-relaxed max-w-2xl mx-auto">
            We trained both types of AI on 30 years of NASA climate satellite data — covering every Indian state from 1995 to 2024 — and measured which one was better at predicting extreme weather events like heatwaves, droughts, and floods.
          </p>
          <p className="text-slate-500 text-base mb-10 max-w-xl mx-auto">
            Short answer: <span className="classical-text font-semibold">classical computers win today</span> — but not because quantum is bad. The reason why is more interesting than the result.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12 text-left">
            <div className="card border-l-4 border-slate-600">
              <div className="text-slate-400 text-xs mb-1">What was being predicted</div>
              <div className="text-white font-semibold text-sm">5 climate conditions</div>
              <div className="text-slate-500 text-xs mt-1">Heatwave · Cold spell · Drought · Flood · Normal — per Indian state, per month</div>
            </div>
            <div className="card border-l-4 border-blue-700">
              <div className="text-slate-400 text-xs mb-1">Classical AI</div>
              <div className="text-white font-semibold text-sm">Trained on full dataset</div>
              <div className="text-slate-500 text-xs mt-1">~7,000 examples · standard laptop-style computation</div>
            </div>
            <div className="card border-l-4 border-violet-700">
              <div className="text-slate-400 text-xs mb-1">Quantum AI</div>
              <div className="text-white font-semibold text-sm">Constrained to 400 examples</div>
              <div className="text-slate-500 text-xs mt-1">IBM quantum computer · 4 qubits · takes hours, not seconds</div>
            </div>
          </div>

          <div className="flex gap-4 justify-center flex-wrap">
            <a href="#comparison" className="inline-block px-6 py-3 bg-violet-700 hover:bg-violet-600 text-white rounded-lg font-medium transition-colors text-sm">
              See what happened →
            </a>
            <a href="/simulator" className="inline-block px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-medium transition-colors text-sm">
              Try the simulator →
            </a>
          </div>
        </div>
      </section>

      {/* COMPARISON */}
      <SectionWrapper id="comparison" dark title="Who got it right more often?" subtitle="Each bar shows how often each model correctly identified the climate condition for a given Indian state and month. Higher is better.">
        <div className="mb-4 card bg-slate-800/60 border-l-4 border-blue-700">
          <p className="text-slate-300 text-sm leading-relaxed">
            <span className="classical-text font-semibold">XGBoost</span> (classical) got the right answer <strong className="text-white">{bestClassicalPct}% of the time</strong> — roughly 6 in every 10 months, for every state. That's the best result overall. The quantum models, working with far less data, got it right about <strong className="text-white">{quantumPct}%</strong> of the time.
          </p>
        </div>
        <div className="mb-6">
          <ModelComparisonChart performance={performance} />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          <WhyCard title="Why is XGBoost winning?" accent="blue">
            XGBoost is a "gradient boosting" method — it builds hundreds of small decision trees and combines their guesses. It's been refined for decades, runs on ordinary hardware, and had access to all 7,000 training examples. It sees the full picture.
          </WhyCard>
          <WhyCard title="Why are quantum models lower?" accent="violet">
            Quantum computers today can only handle tiny problems — so we had to cut the training data from ~7,000 examples down to just 400, and compress 9 measurements into 4. That's like asking a student to learn from 1/17th of the textbook. The result reflects that constraint, not quantum AI's potential.
          </WhyCard>
        </div>
      </SectionWrapper>

      {/* SCALING */}
      <SectionWrapper id="scaling" title="Why couldn't quantum just use more data?" subtitle="Drag the slider. See how long each approach takes as the training set grows.">
        <ScalingSimulator />
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <WhyCard title="Classical AI scales gracefully" accent="blue">
            Classical models get a little slower as you add more data — but it's manageable. Adding 10× more examples might take 2–3× longer. You can train on millions of examples overnight.
          </WhyCard>
          <WhyCard title="Quantum AI has a time explosion problem" accent="violet">
            The quantum kernel method (QSVC) needs to compare every example against every other example using a real quantum circuit. Double the data → 4× the work. At the full dataset size, it would take roughly 82 hours. That's why quantum was limited to 400 examples — it's a hardware maturity limit, not a design flaw.
          </WhyCard>
        </div>
      </SectionWrapper>

      {/* FEATURES */}
      <SectionWrapper id="features" dark title="Fitting climate into a quantum computer" subtitle="Quantum computers can only process a few inputs at a time. Here's how 9 climate measurements get compressed into 4 rotation angles on a quantum chip.">
        <div className="mb-6 card bg-slate-800/60 border-l-4 border-violet-700">
          <p className="text-slate-300 text-sm leading-relaxed">
            Think of a qubit like a dial that can point anywhere on a sphere. To encode a climate measurement, we rotate the dial by a specific angle. But we only have 4 dials — so 9 original measurements (temperature, humidity, wind, pressure, etc.) must be compressed into 4 without losing too much information. This compression kept <strong className="text-white">94% of the useful signal</strong>.
          </p>
        </div>
        <FeatureFlowDiagram dataEfficiency={dataEfficiency} />
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-center text-sm">
          <div className="card">
            <div className="text-slate-400 text-xs mb-1">Measurements going in</div>
            <div className="text-white font-mono text-2xl font-bold">9</div>
            <div className="text-slate-500 text-xs">temperature, humidity, wind, pressure, precipitation, cloud cover…</div>
          </div>
          <div className="card border-l-4 border-violet-700">
            <div className="text-slate-400 text-xs mb-1">After compression</div>
            <div className="text-violet-300 font-mono text-2xl font-bold">4</div>
            <div className="text-slate-500 text-xs">94% of the information preserved — the least important detail discarded</div>
          </div>
          <div className="card">
            <div className="text-slate-400 text-xs mb-1">Each becomes</div>
            <div className="text-teal-300 font-mono text-2xl font-bold">an angle</div>
            <div className="text-slate-500 text-xs">a qubit rotation on a real superconducting chip</div>
          </div>
        </div>
      </SectionWrapper>

      {/* KERNEL */}
      <SectionWrapper id="kernel" title="How quantum AI sees similarity" subtitle="Before classifying climate, the quantum model builds a map of how similar every pair of months is to each other.">
        <div className="mb-6 card bg-slate-800/60 border-l-4 border-blue-700">
          <p className="text-slate-300 text-sm leading-relaxed">
            The quantum model's strategy is: "If two months have very similar quantum states, they're probably the same kind of weather." The heatmap below shows those similarity scores. <strong className="text-white">Bright = very similar. Dark = very different.</strong> The diagonal is always bright because each month is identical to itself.
          </p>
        </div>
        <KernelMatrixChart kernelStats={kernelStats} />
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <WhyCard title="Simulator vs real hardware" accent="blue">
            The left heatmap was computed on a regular computer simulating quantum circuits — it shows crisp, varied similarity scores. Real quantum chips introduce errors (like static noise on a radio), which compress all values toward the middle, making it harder to tell months apart.
          </WhyCard>
          <WhyCard title="Why this matters for accuracy" accent="violet">
            A good similarity map has high contrast — some pairs very similar, others very different. Noise from real hardware flattens this contrast. When everything looks "kind of similar", the model can't make confident decisions, which directly hurts its accuracy.
          </WhyCard>
        </div>
      </SectionWrapper>

      {/* OPTIMIZATION */}
      <SectionWrapper id="optimization" dark title="Did the models actually learn?" subtitle="Training an AI means adjusting its internal settings until it gets better. Here's what that process looked like for the classical neural network vs. the quantum model.">
        <VQCOptimizationChart optimization={optimization} />
      </SectionWrapper>

      {/* PIPELINE */}
      <SectionWrapper id="pipeline" title="How long did everything take?" subtitle="From raw data to final prediction — the time cost of each approach.">
        <PipelineBreakdownChart efficiency={efficiency} />
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <WhyCard title="Classical: fast enough to experiment" accent="blue">
            Classical models trained in under a minute. That means you can try dozens of configurations, tune settings, and rerun experiments easily. Fast feedback is essential for building good AI.
          </WhyCard>
          <WhyCard title="Quantum: 46 minutes for one run" accent="violet">
            The quantum kernel method took 46 minutes just to compute similarities between 400 training examples. That's before any prediction. Slow feedback cycles make it hard to iterate — which is one reason quantum AI is harder to develop right now.
          </WhyCard>
        </div>
      </SectionWrapper>

      {/* INSIGHTS */}
      <SectionWrapper id="insights" dark title="So... is quantum AI useless?" subtitle="No — but the reason it lost here reveals exactly what it needs to win.">
        <div className="mb-6 card bg-slate-800/60 border-l-4 border-amber-700">
          <p className="text-slate-300 text-sm leading-relaxed">
            Classical AI won every metric in this experiment. But this was never a fair fight — and understanding why tells you more than the scoreboard does.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <WhyCard title="The real constraint: time, not intelligence" accent="violet">
            Quantum AI didn't lose because the approach is wrong. It lost because today's quantum computers are slow and noisy — like early transistor computers in the 1950s. The algorithm ran on a real IBM quantum chip in New York. It just couldn't process enough data to compete yet.
          </WhyCard>
          <WhyCard title="17× less data is a huge handicap" accent="violet">
            Classical models trained on ~7,000 examples. Quantum was limited to 400. If you gave a human student 17× fewer practice problems before an exam, you'd expect them to score lower too. The data gap alone explains most of the accuracy gap.
          </WhyCard>
          <WhyCard title="The quantum model never actually learned" accent="amber">
            A software bug in the quantum training stack meant the model's settings were never adjusted at all — it made predictions using random initial values, like guessing without studying. This is an honest result from the frontier of quantum software development, where tools break in unexpected ways.
          </WhyCard>
          <WhyCard title="What quantum AI is actually promising" accent="blue">
            Quantum computers can represent exponentially more complex patterns than classical ones — in theory. For problems where classical AI genuinely hits a ceiling (drug discovery, materials science, cryptography), quantum may eventually provide a real advantage. Climate tabular data isn't that problem yet.
          </WhyCard>
          <WhyCard title="This is what the frontier looks like" accent="slate">
            Real research isn't clean. Quotas run out (we got 2 of 20 planned hardware rows before IBM's monthly limit was exhausted). Software breaks at version boundaries. Models don't converge. This project shows that honestly — including all the failures — because that's what working at the edge of a new technology actually looks like.
          </WhyCard>
          <WhyCard title="What would change the outcome?" accent="slate">
            More qubits, less noise, and faster circuit execution. When quantum hardware matures to the point where you can train on full datasets without waiting hours per run, the comparison becomes more interesting. That's likely 5–15 years away for problems like this one.
          </WhyCard>
        </div>
        <div className="mt-10 text-center border-t border-slate-800 pt-8">
          <p className="text-slate-400 text-sm max-w-2xl mx-auto leading-relaxed">
            Data: NASA MERRA-2 satellite reanalysis · 1995–2024 · 28 Indian states<br />
            Quantum hardware: IBM ibm_fez · 127-qubit superconducting chip · New York
          </p>
          <p className="text-slate-600 text-xs mt-4">
            This project compares computational regimes, not just performance.
          </p>
        </div>
      </SectionWrapper>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/simulator" element={<Simulator />} />
      </Routes>
    </BrowserRouter>
  )
}
