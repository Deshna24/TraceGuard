import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Shield } from 'lucide-react';
import { TraceGuardProvider } from './hooks/useTraceGuard';
import { Dashboard } from './pages/Dashboard';
import { TrajectoryAnalysis } from './pages/TrajectoryAnalysis';
import { ModelsPage } from './pages/ModelsPage';
import { PolicyPage } from './pages/PolicyPage';
import { Evidence } from './pages/Evidence';
import { SystemHealthPage } from './pages/SystemHealthPage';
import { Architecture } from './pages/Architecture';
import { About } from './pages/About';

function Navigation() {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="border-b border-gray-800 bg-gray-950/80 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-6">
            <Link to="/" className="flex items-center gap-2.5">
              <Shield className="w-6 h-6 text-blue-500" />
              <div>
                <span className="text-white font-extrabold text-base tracking-widest font-mono">TRACEGUARD-X</span>
                <span className="block text-[10px] text-gray-400 font-mono tracking-tight">Uncertainty-Aware Agent Security</span>
              </div>
            </Link>

            <div className="hidden md:flex items-center space-x-1 pl-4 border-l border-gray-800 text-xs font-mono">
              <Link
                to="/"
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  isActive('/') ? 'text-white bg-blue-900/30 border border-blue-700/50' : 'text-gray-400 hover:text-white'
                }`}
              >
                Security Overview
              </Link>
              <Link
                to="/analyze"
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  isActive('/analyze') ? 'text-white bg-blue-900/30 border border-blue-700/50' : 'text-gray-400 hover:text-white'
                }`}
              >
                Trajectory Analysis
              </Link>
              <Link
                to="/models"
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  isActive('/models') ? 'text-white bg-blue-900/30 border border-blue-700/50' : 'text-gray-400 hover:text-white'
                }`}
              >
                Model Registry
              </Link>
              <Link
                to="/policy"
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  isActive('/policy') ? 'text-white bg-blue-900/30 border border-blue-700/50' : 'text-gray-400 hover:text-white'
                }`}
              >
                Policy Engine
              </Link>
              <Link
                to="/evidence"
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  isActive('/evidence') ? 'text-white bg-blue-900/30 border border-blue-700/50' : 'text-gray-400 hover:text-white'
                }`}
              >
                Evaluation & Evidence
              </Link>
              <Link
                to="/system"
                className={`px-3 py-1.5 rounded-md transition-colors ${
                  isActive('/system') ? 'text-white bg-blue-900/30 border border-blue-700/50' : 'text-gray-400 hover:text-white'
                }`}
              >
                System Health
              </Link>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-950/40 border border-emerald-800 text-[11px] font-mono text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> Local Node Active
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
}

function App() {
  return (
    <TraceGuardProvider>
      <Router>
        <div className="min-h-screen bg-background text-gray-100 flex flex-col font-sans">
          <Navigation />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analyze" element={<TrajectoryAnalysis />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/policy" element={<PolicyPage />} />
              <Route path="/evidence" element={<Evidence />} />
              <Route path="/system" element={<SystemHealthPage />} />
              <Route path="/architecture" element={<Architecture />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </main>
        </div>
      </Router>
    </TraceGuardProvider>
  );
}

export default App;
