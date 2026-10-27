import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldCheck, ShieldAlert, Clock, ShieldX, HelpCircle, Eye } from 'lucide-react';

interface InterceptionPipelineProps {
  gateDetails: any;
  toolExecutionCounts: Record<string, number>;
  status: string;
  currentTool: string | null;
  probabilities: any;
}

export const InterceptionPipeline: React.FC<InterceptionPipelineProps> = ({ 
  gateDetails, 
  toolExecutionCounts, 
  status, 
  currentTool,
  probabilities
}) => {
  if (status !== 'ACTING' && status !== 'WAITING' && status !== 'BLOCKED' && !gateDetails && !currentTool) {
    return null;
  }

  const isBlocked = status === 'BLOCKED' || (gateDetails && gateDetails.decision === 'BLOCK');
  const isAbstained = gateDetails && gateDetails.decision === 'ABSTAIN';
  const isEvaluating = status === 'ACTING';
  
  const displayTool = gateDetails?.tool || currentTool;
  const executionCount = displayTool ? (toolExecutionCounts[displayTool] || 0) : 0;

  const modelLabel = probabilities?.model_architecture 
    ? `TRACEGUARD-X ${String(probabilities.model_architecture).toUpperCase()}` 
    : 'TRACEGUARD LSTM';

  return (
    <div className={`glass-panel p-6 flex flex-col gap-6 border-2 transition-colors duration-500 ${
      isBlocked ? 'bg-red-950/20 border-red-500/50' : 
      isAbstained ? 'bg-amber-950/25 border-amber-500/60' :
      isEvaluating ? 'bg-amber-950/20 border-amber-500/50' :
      'bg-blue-950/20 border-blue-500/50'
    }`}>
      <div className="flex justify-between items-center border-b border-gray-800 pb-2">
        <h2 className="text-xl font-bold flex items-center gap-2">
          {isBlocked ? <ShieldAlert className="w-6 h-6 text-red-500" /> : 
           isAbstained ? <HelpCircle className="w-6 h-6 text-amber-500" /> :
           isEvaluating ? <Clock className="w-6 h-6 text-amber-500 animate-pulse" /> : 
           <ShieldCheck className="w-6 h-6 text-emerald-500" />}
          <span className={
            isBlocked ? "text-red-500" : 
            isAbstained ? "text-amber-400" :
            isEvaluating ? "text-amber-500" : 
            "text-emerald-500"
          }>
            REAL-TIME SECURITY INTERCEPT
          </span>
        </h2>
        {isBlocked && (
          <span className="bg-red-500 text-white text-xs font-bold px-2 py-1 rounded animate-pulse">
            INTERCEPTED BEFORE EXECUTION (BLOCKED)
          </span>
        )}
        {isAbstained && (
          <span className="bg-amber-500 text-black text-xs font-bold px-2 py-1 rounded animate-pulse">
            ESCALATED TO OPERATOR (ABSTAINED)
          </span>
        )}
      </div>

      <div className="flex flex-col items-center justify-center space-y-4 py-4 font-mono text-sm">
        
        {/* Step 1: Agent Proposal */}
        <motion.div 
          initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center"
        >
          <div className="text-gray-400 mb-1">AGENT PROPOSED ACTION</div>
          <div className="bg-gray-900 border border-gray-700 px-6 py-2 rounded text-blue-300 font-bold text-lg">
            {displayTool || '...'}
          </div>
          <div className="text-gray-400 text-xs mt-2 italic">
            {isEvaluating ? 'Asking the local AI model what to do next...' : 'Local AI model proposed an action'}
          </div>
        </motion.div>

        <div className="h-6 w-0.5 bg-gray-600"></div>

        {/* Step 2: TraceGuard Check */}
        <motion.div 
          initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
          className="flex flex-col items-center w-full"
        >
          <div className="text-gray-400 mb-1 flex items-center gap-2">
            SECURITY ANALYSIS COMPARISON
          </div>
          <div className="flex gap-4 w-full max-w-2xl">
            {/* Baseline */}
            <div className={`p-3 rounded border w-1/2 flex flex-col justify-center items-center ${
              isEvaluating ? 'bg-amber-900/20 border-amber-500/50 text-amber-300' :
              (probabilities && probabilities.baseline_p_hijacked >= 0.5) ? 'bg-red-900/20 border-red-500/50 text-red-300' :
              'bg-emerald-900/20 border-emerald-500/50 text-emerald-300'
            }`}>
              <div className="text-[10px] font-bold mb-1 opacity-70 text-center uppercase tracking-wider">Single-Step Baseline</div>
              {isEvaluating ? (
                <span className="flex items-center gap-2 animate-pulse text-xs"><Clock className="w-3 h-3" /> EVALUATING...</span>
              ) : probabilities ? (
                <div className="text-center">
                  <div className="font-mono text-xs">P(HIJACKED) = {(probabilities.baseline_p_hijacked * 100).toFixed(1)}%</div>
                  <div className="font-bold mt-1 text-sm">{probabilities.baseline_p_hijacked >= 0.5 ? 'HIJACKED — BLOCKED' : 'BENIGN (ALLOW)'}</div>
                </div>
              ) : <span className="text-xs">COMPLETE</span>}
            </div>

            {/* Model (LSTM, GRU, Attention, or MultiTask) */}
            <div className={`p-3 rounded border w-1/2 flex flex-col justify-center items-center ${
              isEvaluating ? 'bg-amber-900/20 border-amber-500/50 text-amber-300' :
              isBlocked ? 'bg-red-900/20 border-red-500/50 text-red-300' :
              isAbstained ? 'bg-amber-900/30 border-amber-500/50 text-amber-300' :
              'bg-emerald-900/20 border-emerald-500/50 text-emerald-300'
            }`}>
              <div className="text-[10px] font-bold mb-1 opacity-70 text-center flex items-center justify-center gap-1 uppercase tracking-wider">
                {modelLabel} <span className="bg-blue-500/20 text-blue-300 px-1 rounded ml-1">FULL TRAJECTORY</span>
              </div>
              {isEvaluating ? (
                <span className="flex items-center gap-2 animate-pulse text-xs"><Clock className="w-3 h-3" /> EVALUATING...</span>
              ) : probabilities ? (
                <div className="text-center">
                  <div className="font-mono text-xs">P(HIJACKED) = {(probabilities.p_hijacked * 100).toFixed(1)}%</div>
                  <div className="font-bold mt-1 text-sm">
                    {isBlocked ? 'HIJACKED — BLOCKED' : isAbstained ? 'AMBIGUOUS — ABSTAIN' : 'ALLOW'}
                  </div>
                  {probabilities.warning_score !== undefined && probabilities.warning_score !== null && (
                    <div className="text-[10px] text-amber-400 mt-1">
                      Imminent Dev Warn: {(Number(probabilities.warning_score) * 100).toFixed(1)}%
                    </div>
                  )}
                  {probabilities.uncertainty_score !== undefined && probabilities.uncertainty_score !== null && (
                    <div className="text-[10px] text-purple-300 mt-0.5">
                      Entropy H(P): {(Number(probabilities.uncertainty_score) * 100).toFixed(1)}%
                    </div>
                  )}
                </div>
              ) : <span className="text-xs">COMPLETE</span>}
            </div>
          </div>
          
          {/* Step Attention Weights Display */}
          {probabilities?.attention_weights && Array.isArray(probabilities.attention_weights) && probabilities.attention_weights.length > 0 && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="w-full max-w-2xl mt-3 bg-gray-900/90 border border-purple-500/40 rounded p-3 text-xs font-mono">
              <div className="flex items-center justify-between text-purple-300 mb-2 font-bold">
                <span className="flex items-center gap-1.5"><Eye className="w-3.5 h-3.5" /> Temporal Attention Distribution</span>
                <span className="text-[10px] text-gray-500 font-normal">Diagnostic pooling only (non-causal)</span>
              </div>
              <div className="flex gap-2 items-center flex-wrap">
                {probabilities.attention_weights.map((w: number, i: number) => (
                  <div key={i} className="flex-1 min-w-[70px] bg-gray-950 p-1.5 rounded border border-gray-800 text-center">
                    <div className="text-[10px] text-gray-400">Step {i + 1}</div>
                    <div className="text-xs font-bold text-purple-300">{(w * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {probabilities && probabilities.baseline_p_hijacked < 0.5 && isBlocked && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="w-full max-w-2xl mt-4 bg-purple-900/20 border border-purple-500/50 rounded p-3 text-xs text-purple-200">
              <div className="font-bold mb-1 flex items-center gap-2 text-purple-400">
                <ShieldCheck className="w-4 h-4" /> WHY DOES THIS NEED TEMPORAL DEEP LEARNING?
              </div>
              <p>
                A single message or action can look completely innocent in isolation. The Single-Step Baseline misses the attack because the malicious intent is spread across multiple turns. The temporal sequence model reads the agent's <b>entire behavior in order</b> and catches the drift. The disagreement above proves the temporal model's value.
              </p>
            </motion.div>
          )}
          
        </motion.div>

        <div className="h-6 w-0.5 bg-gray-600"></div>

        {/* Step 3: Gate Decision */}
        <AnimatePresence mode="wait">
          {!isEvaluating && gateDetails && (
            <motion.div 
              key="decision"
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center"
            >
              <div className="text-gray-400 mb-1">SAFETY GATE DECISION</div>
              <div className={`px-8 py-2 rounded font-bold text-xl ${
                isBlocked ? 'bg-red-500 text-white shadow-[0_0_15px_rgba(239,68,68,0.5)]' : 
                isAbstained ? 'bg-amber-500 text-black shadow-[0_0_15px_rgba(245,158,11,0.5)]' :
                'bg-emerald-500 text-white shadow-[0_0_15px_rgba(16,185,129,0.5)]'
              }`}>
                {gateDetails.decision}
              </div>
              <div className="text-gray-400 text-xs mt-2 italic text-center max-w-md">
                {isBlocked 
                  ? 'Decision: blocking this action (pre-action safety boundary)' 
                  : isAbstained 
                  ? 'Decision: escalated / abstained due to elevated risk or high uncertainty'
                  : 'Decision: allowing this action'}
              </div>
              {gateDetails.reason && (
                <div className="text-[11px] text-amber-300/80 mt-1 font-mono text-center max-w-lg">
                  {gateDetails.reason}
                </div>
              )}
            </motion.div>
          )}
          {isEvaluating && (
            <motion.div key="evaluating-gate" className="flex flex-col items-center opacity-50">
               <div className="text-gray-400 mb-1">SAFETY GATE DECISION</div>
               <div className="px-8 py-2 rounded bg-gray-800 text-gray-500 font-bold text-xl border border-gray-700">
                 PENDING
               </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="h-6 w-0.5 bg-gray-600"></div>

        {/* Step 4: Tool Execution */}
        <AnimatePresence mode="wait">
          {!isEvaluating && gateDetails && (
            <motion.div 
              key="execution"
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="flex flex-col items-center w-full"
            >
              <div className="text-gray-400 mb-1">TOOL EXECUTION</div>
              {(isBlocked || isAbstained) ? (
                <div className={`bg-gray-950 border-2 p-4 rounded w-full max-w-md text-center flex flex-col gap-2 ${
                  isBlocked ? 'border-red-900/50' : 'border-amber-900/50'
                }`}>
                  <div className={`flex items-center justify-center gap-2 font-bold text-lg ${
                    isBlocked ? 'text-red-400' : 'text-amber-400'
                  }`}>
                    <ShieldX className="w-5 h-5" /> NEVER STARTED {isAbstained ? '(ESCALATED)' : '(BLOCKED)'}
                  </div>
                  <div className="w-full bg-red-950/50 h-2 rounded overflow-hidden mt-1 relative">
                    <div className="absolute top-0 left-0 h-full w-full bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(239,68,68,0.2)_10px,rgba(239,68,68,0.2)_20px)]"></div>
                  </div>
                </div>
              ) : (
                <div className="bg-gray-950 border border-emerald-900/50 p-4 rounded w-full max-w-md text-center">
                  <div className="text-emerald-400 font-bold mb-2">EXECUTING</div>
                  <div className="w-full bg-emerald-900/20 h-2 rounded overflow-hidden">
                    <div className="bg-emerald-500 h-full animate-pulse w-full"></div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
          {isEvaluating && (
            <motion.div key="evaluating-exec" className="flex flex-col items-center opacity-30 w-full">
               <div className="text-gray-400 mb-1">TOOL EXECUTION</div>
               <div className="bg-gray-950 border border-gray-800 p-4 rounded w-full max-w-md text-center text-gray-600">
                 AWAITING SECURITY DECISION
               </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>

      {/* Proof Panel */}
      <div className="bg-gray-950 p-4 rounded border border-gray-800 flex justify-between items-center text-sm mt-2">
         <div className="text-gray-400 flex items-center gap-2">
            TOOL EXECUTION MONITOR
         </div>
         <div className="flex items-center gap-4">
            <span className="text-gray-500">{displayTool}</span>
            <span className={`font-mono font-bold px-3 py-1 rounded ${
               isBlocked ? 'bg-red-950 text-red-400 border border-red-900/50' : 
               isAbstained ? 'bg-amber-950 text-amber-400 border border-amber-900/50' :
               'bg-gray-900 text-gray-300'
            }`}>
              Count: {executionCount}
            </span>
         </div>
      </div>
    </div>
  );
};
