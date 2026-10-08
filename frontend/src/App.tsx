import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <nav className="border-b border-border bg-panel">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-8">
                <Link to="/" className="text-blue-400 font-bold text-xl tracking-wider">TRACEGUARD</Link>
                <div className="hidden md:block">
                  <div className="flex items-baseline space-x-4">
                    <Link to="/" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Live Demo</Link>
                    <Link to="/architecture" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Architecture</Link>
                    <Link to="/attack-lab" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Attack Lab</Link>
                    <Link to="/evidence" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">Research Evidence</Link>
                    <Link to="/about" className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium">About</Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </nav>
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/architecture" element={<div className="p-8 text-white">Architecture (Coming Soon)</div>} />
            <Route path="/attack-lab" element={<div className="p-8 text-white">Attack Lab (Coming Soon)</div>} />
            <Route path="/evidence" element={<div className="p-8 text-white">Research Evidence (Coming Soon)</div>} />
            <Route path="/about" element={<div className="p-8 text-white">About (Coming Soon)</div>} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
