import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/layout/Navbar'
import Simulator from './pages/Simulator'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Simulator />} />
        <Route path="/simulator" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
