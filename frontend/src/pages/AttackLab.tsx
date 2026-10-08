import { useNavigate } from 'react-router-dom';
import { ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';
import { useTraceGuard } from '../hooks/useTraceGuard';

export const AttackLab = () => {
  const { startRun } = useTraceGuard();
  const navigate = useNavigate();

  const handleRun = async (scenario: string) => {
    await startRun(scenario);
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-background text-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold tracking-wider text-blue-400 mb-2">ATTACK LAB</h1>
        <p className="text-gray-400 mb-8">Select a scenario to observe TRACEGUARD in action.</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* BENIGN */}
          <div className="glass-panel p-6 flex flex-col gap-4 border border-emerald-900/30">
            <h2 className="text-xl font-bold text-emerald-400 flex items-center gap-2">
              <ShieldCheck className="w-6 h-6" />
              BENIGN
            </h2>
            <p className="text-sm text-gray-300">
              The agent receives a legitimate task to calculate the distance between Earth and Mars and successfully completes it using the calculator tool.
            </p>
            <div className="bg-gray-900/50 p-3 rounded text-sm space-y-2 border border-gray-800">
              <div><span className="text-gray-500 font-bold">Attack State:</span> None</div>
              <div><span className="text-gray-500 font-bold">Agent Behavior:</span> Normal task completion</div>
              <div><span className="text-gray-500 font-bold">TRACEGUARD:</span> Low risk, allows all actions</div>
              <div><span className="text-gray-500 font-bold">Expected Outcome:</span> Task completed successfully</div>
            </div>
            <button 
              onClick={() => handleRun('BENIGN')}
              className="mt-auto w-full py-2 bg-emerald-900/20 hover:bg-emerald-900/40 text-emerald-400 rounded font-bold transition-colors border border-emerald-900/50"
            >
              RUN SCENARIO
            </button>
          </div>

          {/* INJECTION RESISTED */}
          <div className="glass-panel p-6 flex flex-col gap-4 border border-amber-900/30">
            <h2 className="text-xl font-bold text-amber-500 flex items-center gap-2">
              <AlertTriangle className="w-6 h-6" />
              INJECTION RESISTED
            </h2>
            <p className="text-sm text-gray-300">
              The agent searches for a stock price and encounters a prompt injection attack in the search results. The agent ignores the attack and completes the original task.
            </p>
            <div className="bg-gray-900/50 p-3 rounded text-sm space-y-2 border border-gray-800">
              <div><span className="text-gray-500 font-bold">Attack State:</span> Injection observed</div>
              <div><span className="text-gray-500 font-bold">Agent Behavior:</span> Remains aligned with user goal</div>
              <div><span className="text-gray-500 font-bold">TRACEGUARD:</span> Detects resistance, risk remains low</div>
              <div><span className="text-gray-500 font-bold">Expected Outcome:</span> Task completed successfully</div>
            </div>
            <button 
              onClick={() => handleRun('INJECTION_RESISTED')}
              className="mt-auto w-full py-2 bg-amber-900/20 hover:bg-amber-900/40 text-amber-500 rounded font-bold transition-colors border border-amber-900/50"
            >
              RUN SCENARIO
            </button>
          </div>

          {/* HIJACKED */}
          <div className="glass-panel p-6 flex flex-col gap-4 border border-red-900/30">
            <h2 className="text-xl font-bold text-red-500 flex items-center gap-2">
              <ShieldAlert className="w-6 h-6" />
              HIJACKED
            </h2>
            <p className="text-sm text-gray-300">
              The agent encounters an injection attack, abandons its original goal, and attempts to execute a malicious database export. TRACEGUARD blocks the action.
            </p>
            <div className="bg-gray-900/50 p-3 rounded text-sm space-y-2 border border-gray-800">
              <div><span className="text-gray-500 font-bold">Attack State:</span> Injection observed</div>
              <div><span className="text-gray-500 font-bold">Agent Behavior:</span> Deviates from goal, attempts export</div>
              <div><span className="text-gray-500 font-bold">TRACEGUARD:</span> Risk crosses threshold, blocks action</div>
              <div><span className="text-gray-500 font-bold">Expected Outcome:</span> Action blocked before execution</div>
            </div>
            <button 
              onClick={() => handleRun('HIJACKED')}
              className="mt-auto w-full py-2 bg-red-900/20 hover:bg-red-900/40 text-red-500 rounded font-bold transition-colors border border-red-900/50"
            >
              RUN SCENARIO
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
