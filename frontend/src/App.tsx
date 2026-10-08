
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { Architecture } from './pages/Architecture';
import { AttackLab } from './pages/AttackLab';
import { Evidence } from './pages/Evidence';
import { About } from './pages/About';

function Navigation() {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;

  return (
    <nav className="border-b border-border bg-panel">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <Link to="/" className="text-blue-400 font-bold text-xl tracking-wider">TRACEGUARD</Link>
            <div className="hidden md:block">
              <div className="flex items-baseline space-x-4">
                <Link to="/" className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${isActive('/') ? 'text-white bg-blue-900/30' : 'text-gray-300 hover:text-white'}`}>Live Demo</Link>
                <Link to="/architecture" className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${isActive('/architecture') ? 'text-white bg-blue-900/30' : 'text-gray-300 hover:text-white'}`}>Architecture</Link>
                <Link to="/attack-lab" className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${isActive('/attack-lab') ? 'text-white bg-blue-900/30' : 'text-gray-300 hover:text-white'}`}>Attack Lab</Link>
                <Link to="/evidence" className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${isActive('/evidence') ? 'text-white bg-blue-900/30' : 'text-gray-300 hover:text-white'}`}>Research Evidence</Link>
                <Link to="/about" className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${isActive('/about') ? 'text-white bg-blue-900/30' : 'text-gray-300 hover:text-white'}`}>About</Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <Navigation />
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/architecture" element={<Architecture />} />
            <Route path="/attack-lab" element={<AttackLab />} />
            <Route path="/evidence" element={<Evidence />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
