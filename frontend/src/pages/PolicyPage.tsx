import React, { useState } from 'react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Scale, Calculator } from 'lucide-react';

interface PolicyEvalResult {
  decision: 'ALLOW' | 'BLOCK' | 'ABSTAIN';
  reason: string;
  action_permitted: boolean;
  requires_operator_review: boolean;
  threshold_block: number;
  threshold_abstain?: number;
  entropy_tau?: number;
}

export const PolicyPage: React.FC = () => {
  const [pHijacked, setPHijacked] = useState<number>(0.38);
  const [entropy, setEntropy] = useState<number>(0.82);
  const [gateType, setGateType] = useState<string>('uncertainty_aware');
  const [evalResult, setEvalResult] = useState<PolicyEvalResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);

  const handleEvaluate = async () => {
    setIsEvaluating(true);
    try {
      const res = await fetch('http://localhost:8000/api/policy/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          p_hijacked: pHijacked,
          entropy: entropy,
          gate_type: gateType,
        }),
      });
      const data: PolicyEvalResult = await res.json();
      setEvalResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-gray-100 p-6 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-wider text-blue-400">SAFETY GATE & POLICY ENGINE</h1>
        <p className="text-gray-400 text-sm mt-1">
          Pre-action enforcement boundaries, uncertainty-aware 3-way decision policies, and fail-closed operator escalation.
        </p>
      </div>

      {/* Policy Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Mode 1: Standard Binary Gate */}
        <div className="glass-panel p-6 rounded-lg border border-gray-800 bg-gray-950/40 space-y-4">
          <div className="flex items-center gap-2">
            <Scale className="w-5 h-5 text-blue-400" />
            <h2 className="text-base font-bold text-gray-200 font-mono">1. Standard Binary Gate (Baseline)</h2>
          </div>
          <p className="text-xs text-gray-400 leading-relaxed">
            The canonical TRACEGUARD baseline evaluates a fixed inclusive threshold on P(HIJACKED):
          </p>
          <div className="p-3 bg-gray-900 border border-gray-800 rounded font-mono text-xs space-y-1.5">
            <div className="text-red-400">• P(HIJACKED) ≥ 0.50 ➔ BLOCK (Execution Denied)</div>
            <div className="text-emerald-400">• P(HIJACKED) &lt; 0.50 ➔ ALLOW (Execution Permitted)</div>
          </div>
          <div className="p-3 bg-red-950/20 border border-red-900/40 rounded text-[11px] text-gray-400">
            <strong className="text-red-300">Limitation:</strong> Susceptible to false negatives when precursor risk or behavioral ambiguity falls in the [0.25, 0.50) band.
          </div>
        </div>

        {/* Mode 2: Uncertainty-Aware 3-Way Gate */}
        <div className="glass-panel p-6 rounded-lg border border-amber-900/40 bg-gray-950/40 space-y-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            <h2 className="text-base font-bold text-amber-300 font-mono">2. Uncertainty-Aware 3-Way Gate (TRACEGUARD-X)</h2>
          </div>
          <p className="text-xs text-gray-400 leading-relaxed">
            Incorporate predictive uncertainty (normalized Shannon entropy H) and a dedicated abstain escalation band:
          </p>
          <div className="p-3 bg-gray-900 border border-gray-800 rounded font-mono text-xs space-y-1.5">
            <div className="text-red-400">• P(HIJACKED) ≥ 0.50 ➔ BLOCK (Action Denied)</div>
            <div className="text-amber-400">• 0.25 ≤ P(HIJACKED) &lt; 0.50 OR H(P) ≥ 0.70 ➔ ABSTAIN</div>
            <div className="text-emerald-400">• P(HIJACKED) &lt; 0.25 AND H(P) &lt; 0.70 ➔ ALLOW</div>
          </div>
          <div className="p-3 bg-amber-950/20 border border-amber-900/40 rounded text-[11px] text-gray-400">
            <strong className="text-amber-300">Fail-Closed Guarantee:</strong> Both BLOCK and ABSTAIN decisions strictly prevent tool execution (execution count = 0).
          </div>
        </div>
      </div>

      {/* Interactive Policy Calculator */}
      <div className="glass-panel p-6 rounded-lg border border-gray-800 bg-gray-950/50 space-y-5">
        <div className="flex items-center gap-2 border-b border-gray-800 pb-3">
          <Calculator className="w-5 h-5 text-blue-400" />
          <h2 className="text-sm font-bold tracking-wider text-gray-200 uppercase font-mono">
            Interactive Policy Evaluator
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-xs font-mono text-gray-400 uppercase font-bold mb-2">
              P(HIJACKED): <span className="text-blue-300 font-bold">{pHijacked.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={pHijacked}
              onChange={(e) => setPHijacked(parseFloat(e.target.value))}
              className="w-full accent-blue-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-gray-500 font-mono mt-1">
              <span>0.0 (Safe)</span>
              <span>0.25 (Abstain)</span>
              <span>0.50 (Block)</span>
              <span>1.0 (Critical)</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-mono text-gray-400 uppercase font-bold mb-2">
              Normalized Shannon Entropy H: <span className="text-amber-300 font-bold">{entropy.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value={entropy}
              onChange={(e) => setEntropy(parseFloat(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-gray-500 font-mono mt-1">
              <span>0.0 (Certain)</span>
              <span>0.70 (Tau Threshold)</span>
              <span>1.0 (Max Entropy)</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-mono text-gray-400 uppercase font-bold mb-2">Policy Gate Mode</label>
            <select
              value={gateType}
              onChange={(e) => setGateType(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-xs font-mono text-gray-200 focus:outline-none focus:border-blue-500"
            >
              <option value="uncertainty_aware">Uncertainty-Aware 3-Way Gate</option>
              <option value="standard">Standard Binary Gate (0.50)</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleEvaluate}
          disabled={isEvaluating}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded font-bold text-xs font-mono uppercase tracking-wider transition-colors"
        >
          {isEvaluating ? 'Evaluating...' : 'Evaluate Policy Output'}
        </button>

        {evalResult && (
          <div
            className={`p-4 rounded-lg border mt-4 flex items-start gap-3 ${
              evalResult.decision === 'BLOCK'
                ? 'bg-red-950/40 border-red-700 text-red-200'
                : evalResult.decision === 'ABSTAIN'
                ? 'bg-amber-950/40 border-amber-600 text-amber-200'
                : 'bg-emerald-950/40 border-emerald-700 text-emerald-200'
            }`}
          >
            {evalResult.decision === 'BLOCK' ? (
              <ShieldAlert className="w-6 h-6 text-red-400 shrink-0 mt-0.5" />
            ) : evalResult.decision === 'ABSTAIN' ? (
              <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0 mt-0.5" />
            ) : (
              <ShieldCheck className="w-6 h-6 text-emerald-400 shrink-0 mt-0.5" />
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-sm tracking-wider uppercase font-mono">
                  DECISION: {evalResult.decision}
                </span>
                {evalResult.requires_operator_review && (
                  <span className="px-2 py-0.5 rounded bg-amber-500/20 border border-amber-400 text-amber-300 text-[10px] font-bold">
                    OPERATOR ESCALATION
                  </span>
                )}
              </div>
              <p className="text-xs mt-1 leading-relaxed">{evalResult.reason}</p>
              <div className="text-[11px] mt-2 font-mono text-gray-400">
                Action Execution Status:{' '}
                <span className={evalResult.action_permitted ? 'text-emerald-400 font-bold' : 'text-red-400 font-bold'}>
                  {evalResult.action_permitted ? 'PERMITTED' : 'PREVENTED (TOOL NEVER CALLED)'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
