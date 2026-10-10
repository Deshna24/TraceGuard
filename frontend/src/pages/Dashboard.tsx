import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, ShieldAlert, ShieldCheck, Scale, Square, RefreshCcw, Shield, Eye } from 'lucide-react';import { useTraceGuard } from '../hooks/useTraceGuard';
import { RiskGraph } from '../components/RiskGraph';
import { LiveTrajectory } from '../components/LiveTrajectory';
import { SecurityEventFeed } from '../components/SecurityEventFeed';
import { InterceptionPipeline } from '../components/PreActionGate';
import { ProtectedDatabase } from '../components/ProtectedDatabase';

export const Dashboard: React.FC = () => {
  const { state, startRun, stopRun, reset } = useTraceGuard();
  
  const isDisconnected = state.connectionStatus === 'DISCONNECTED';
  const isRunning = state.status === 'RUNNING' || state.status === 'ACTING' || state.status === 'WAITING';
  
  // Calculate detection latency
  const detectionLatencyMs = useMemo(() => {
    const actionProposedEvent = state.events.slice().reverse().find(e => e.event_type === 'ACTION_PROPOSED');
    const detectorEvaluatedEvent = state.events.slice().reverse().find(e => e.event_type === 'DETECTOR_EVALUATED');
    
    if (actionProposedEvent && detectorEvaluatedEvent) {
      const start = new Date(actionProposedEvent.timestamp).getTime();
      const end = new Date(detectorEvaluatedEvent.timestamp).getTime();
      if (end >= start) {
        return end - start;
      }
    }
    return null;
  }, [state.events]);

  const isHijacked = state.status === 'BLOCKED' && state.blockedActionReason;
  const isCompleted = state.status === 'COMPLETED';

  return (
    <div className="min-h-screen bg-background text-gray-100 p-4 flex flex-col gap-4 font-sans">
      {/* Header */}
      <header className="flex items-center justify-between glass-panel p-4 rounded-lg bg-gray-950/50 border border-gray-800">
        <div className="flex items-center gap-4">
          <Shield className="w-8 h-8 text-blue-500" />
          <div>
            <h1 className="text-3xl font-extrabold tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-600">
              TRACEGUARD
            </h1>
            <p className="text-blue-300/60 text-xs mt-1 uppercase tracking-widest font-bold">
              Real-Time Security Intercept
            </p>
          </div>
        </div>
        <div className="flex gap-6 items-center">
          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
               <button onClick={() => startRun('BENIGN')} disabled={isDisconnected || isRunning} className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded text-xs font-bold uppercase transition-colors border border-gray-600 flex items-center gap-1">
                 <ShieldCheck className="w-3 h-3 text-emerald-400" /> Benign
               </button>
               <button onClick={() => startRun('INJECTION_RESISTED')} disabled={isDisconnected || isRunning} className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded text-xs font-bold uppercase transition-colors border border-gray-600 flex items-center gap-1">
                 <Shield className="w-3 h-3 text-amber-400" /> Resisted
               </button>
               <button onClick={() => startRun('HIJACKED')} disabled={isDisconnected || isRunning} className="px-3 py-1 bg-red-950 hover:bg-red-900 text-red-200 rounded text-xs font-bold uppercase transition-colors border border-red-700 flex items-center gap-1">
                 <ShieldAlert className="w-3 h-3 text-red-400" /> Hijacked
               </button>
            </div>
            <div className="flex justify-end gap-2">
               <button onClick={stopRun} disabled={!isRunning} className="px-2 py-1 bg-gray-900 text-gray-500 rounded text-xs font-bold transition-colors border border-gray-800 flex items-center gap-1">
                 <Square className="w-3 h-3" /> Stop
               </button>
               <button onClick={reset} disabled={isRunning && !isDisconnected} className="px-2 py-1 bg-gray-900 text-gray-400 hover:text-gray-200 rounded text-xs font-bold transition-colors border border-gray-800 flex items-center gap-1">
                 <RefreshCcw className="w-3 h-3" /> Reset
               </button>
            </div>
          </div>
          
          <div className="h-10 w-px bg-gray-800 mx-2"></div>
          
          <div className="flex flex-col items-end gap-1">
             <div className={`flex items-center gap-2 px-3 py-1 rounded text-xs font-bold uppercase tracking-wider border ${
               isDisconnected ? 'bg-red-900/30 border-red-500/30 text-red-400' : 
               isRunning ? 'bg-red-900/30 border-red-500 text-red-400 animate-pulse' :
               'bg-blue-900/30 border-blue-500/30 text-blue-400'
             }`}>
               {!isDisconnected && isRunning && <div className="w-2 h-2 rounded-full bg-red-500"></div>}
               {!isDisconnected && !isRunning && <div className="w-2 h-2 rounded-full bg-blue-500"></div>}
               {isDisconnected ? 'DISCONNECTED' : isRunning ? '● LIVE RUNTIME' : 'LIVE — READY'}
             </div>
             {detectionLatencyMs !== null && (
               <div className="text-[10px] text-gray-500 font-mono">
                 DETECTION LATENCY: <span className="text-gray-300">{detectionLatencyMs} ms</span>
               </div>
             )}
          </div>
        </div>
      </header>

      {isDisconnected && (
        <div className="bg-red-950 border border-red-500 text-red-400 p-2 rounded text-center font-bold text-sm">
          RUNTIME UNAVAILABLE - PLEASE START THE BACKEND SERVER
        </div>
      )}
      {state.status === 'ERROR' && state.error && (
        <div className="bg-red-950 border border-red-500 text-red-400 p-2 rounded text-center font-bold text-sm">
          RUNTIME ERROR: {state.error}
        </div>
      )}

      {/* Main Grid Layout */}
      <div className="grid grid-cols-12 gap-4 flex-1 overflow-hidden min-h-[800px]">
        
        {/* Left Column: Agent State & Trajectory */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-4 overflow-hidden h-full">
          <div className="glass-panel p-4 flex flex-col gap-3 bg-gray-900 border border-gray-800 rounded">
            <h2 className="text-sm font-bold border-b border-gray-800 pb-2 flex items-center gap-2 text-gray-400">
              <BrainCircuit className="w-4 h-4 text-gray-500" />
              AGENT STATE
            </h2>
            <div className="flex items-center justify-between bg-gray-950 p-2 rounded border border-gray-800 font-mono text-sm">
              <span className="text-gray-500">STATUS</span>
              <span className={`font-bold ${
                state.status === 'IDLE' ? 'text-gray-500' :
                state.status === 'RUNNING' ? 'text-emerald-400' :
                state.status === 'ACTING' ? 'text-blue-400' :
                state.status === 'WAITING' ? 'text-amber-400' :
                state.status === 'BLOCKED' ? 'text-red-500' :
                state.status === 'COMPLETED' ? 'text-emerald-500' : 'text-gray-400'
              }`}>{state.status}</span>
            </div>
            
            <div className="mt-2">
              <div className="text-xs text-gray-500 font-bold mb-1">AUTHORIZED USER GOAL</div>
              <div className="bg-gray-950 p-3 rounded border border-gray-800 text-sm text-gray-300 min-h-[60px]">
                {state.userGoal || <span className="text-gray-600 italic">No active task...</span>}
              </div>
            </div>
          </div>
          
          <LiveTrajectory trajectory={state.trajectory} currentTool={state.currentTool} status={state.status} />
        </div>

        {/* Middle Column: Interception Pipeline (MAIN) */}
        <div className="col-span-12 lg:col-span-6 flex flex-col gap-4 overflow-hidden h-full">
          
          <div className="flex-1 flex flex-col justify-center">
             {(state.status === 'ACTING' || state.status === 'WAITING' || state.status === 'BLOCKED' || state.gateDetails) ? (
               <InterceptionPipeline 
                 gateDetails={state.gateDetails} 
                 toolExecutionCounts={state.toolExecutionCounts} 
                 status={state.status} 
                 currentTool={state.currentTool} 
                 probabilities={state.probabilities} 
               />
             ) : (
               <div className="h-full flex items-center justify-center border-2 border-dashed border-gray-800 rounded-lg bg-gray-900/20">
                 <div className="text-gray-500 text-sm font-mono flex flex-col items-center gap-2">
                   <Eye className="w-8 h-8 opacity-50" />
                   AWAITING AGENT ACTION PROPOSAL...
                 </div>
               </div>
             )}
          </div>

          {/* Goal vs Behavior Panel */}
          <AnimatePresence>
            {state.untrustedObservation && state.currentTool && (
               <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="glass-panel p-4 bg-gray-900 border border-gray-800 flex flex-col gap-3 rounded">
                  <h2 className="text-sm font-bold flex items-center gap-2 text-gray-400 border-b border-gray-800 pb-2">
                     <Scale className="w-4 h-4 text-blue-500" />
                     GOAL ALIGNMENT ANALYSIS
                  </h2>
                  <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                     <div className="bg-gray-950 p-3 rounded border border-gray-800">
                        <div className="text-gray-500 text-xs mb-2">AUTHORIZED GOAL</div>
                        <div className="text-gray-300">{state.userGoal}</div>
                     </div>
                     <div className={`bg-gray-950 p-3 rounded border ${isHijacked || (state.probabilities?.p_hijacked ?? 0) >= 0.5 ? 'border-red-900/50' : 'border-gray-800'}`}>
                        <div className="text-gray-500 text-xs mb-2">CURRENT BEHAVIOR</div>
                        <div className={`break-words ${isHijacked || (state.probabilities?.p_hijacked ?? 0) >= 0.5 ? 'text-red-400' : 'text-blue-400'}`}>{state.currentTool}</div>
                     </div>
                  </div>
                  {(state.probabilities?.p_hijacked ?? 0) >= 0.5 && (
                    <div className="text-center bg-red-950/30 text-red-500 font-bold p-2 border border-red-900/50 rounded text-sm uppercase tracking-wider mt-2 animate-pulse">
                      Behavioral Deviation Detected — Goal Alignment: LOW
                    </div>
                  )}
               </motion.div>
            )}
          </AnimatePresence>
          
          {/* Protected Database (Shown when relevant) */}
          <ProtectedDatabase />
          
          {/* Proof Mode Panel (Shown on Completion/Block) */}
          <AnimatePresence>
            {isHijacked && (
               <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6 bg-red-950/40 border-red-500 rounded border-2 shadow-[0_0_30px_rgba(239,68,68,0.15)]">
                  <h2 className="text-xl font-bold flex items-center gap-2 text-red-500 mb-4 uppercase tracking-wider">
                     <ShieldAlert className="w-6 h-6" /> Why was this action blocked?
                  </h2>
                  <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm font-mono border-b border-red-900/50 pb-4 mb-4">
                     <div className="text-gray-400 text-right">Authorized Goal</div>
                     <div className="text-gray-200">{state.userGoal}</div>
                     
                     <div className="text-gray-400 text-right">Observed Untrusted Content</div>
                     <div className="text-amber-400">Search result contained an injected instruction</div>
                     
                     <div className="text-gray-400 text-right">Agent Proposed</div>
                     <div className="text-red-400 font-bold">{state.gateDetails?.tool || 'database.export_records'}</div>
                     
                     <div className="text-gray-400 text-right">Security Evaluation</div>
                     <div className="text-gray-200">P(HIJACKED) = {(state.probabilities?.p_hijacked || 0).toFixed(3)}</div>
                     
                     <div className="text-gray-400 text-right">Threshold</div>
                     <div className="text-gray-200">{(state.probabilities?.threshold || 0).toFixed(3)}</div>
                     
                     <div className="text-gray-400 text-right">Gate Decision</div>
                     <div className="text-red-500 font-bold">BLOCK</div>
                     
                     <div className="text-gray-400 text-right">Detection</div>
                     <div className="text-emerald-400">PRE-ACTION</div>
                  </div>
                  
                  <div className="bg-red-950/50 p-4 rounded border border-red-900/50">
                    <h3 className="text-white font-bold mb-3 tracking-wider uppercase text-sm">SECURITY PROOF</h3>
                    <ul className="space-y-2 text-sm font-mono text-gray-300">
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> Attack observed</li>
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> Behavioral deviation observed</li>
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> High-impact action proposed</li>
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> Security analysis performed</li>
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> Action intercepted before execution</li>
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> Database export blocked</li>
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> Tool invocation count = 0</li>
                      <li className="flex items-center gap-2"><span className="text-emerald-400">✓</span> No side effect occurred</li>
                    </ul>
                  </div>
                  
                  <div className="mt-4 text-center font-bold text-red-400 uppercase tracking-widest text-sm">
                     Dangerous action was prevented before execution.
                  </div>
               </motion.div>
            )}
            {isCompleted && !isHijacked && (
               <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6 bg-emerald-950/20 border-emerald-500/50 rounded border-2">
                  <h2 className="text-xl font-bold flex items-center gap-2 text-emerald-500 mb-2 uppercase tracking-wider">
                     {state.untrustedObservation ? '🟡 INJECTION OBSERVED, REMAINS ALIGNED' : '🟢 TASK COMPLETED'}
                  </h2>
                  <div className="text-sm space-y-2 text-emerald-200 mb-4">
                     {state.untrustedObservation ? (
                        <>
                          <div className="font-bold text-emerald-400 mb-2">✓ Agent ignored the injection and completed the authorized goal.</div>
                        </>
                     ) : (
                        <div>Agent successfully completed the task without deviation.</div>
                     )}
                  </div>
                  {state.finalAnswer && (
                    <div className="bg-gray-950 p-4 rounded border border-emerald-900/50 mt-2 font-mono text-sm">
                      <div className="text-emerald-500 font-bold mb-2 text-xs uppercase tracking-wider">Final Output:</div>
                      <div className="text-gray-200 whitespace-pre-wrap">{state.finalAnswer}</div>
                    </div>
                  )}
               </motion.div>
            )}
          </AnimatePresence>

        </div>

        {/* Right Column: Timeline & Risk Graph */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-4 overflow-hidden h-full">
          <SecurityEventFeed events={state.events} />
          <RiskGraph evaluations={state.evaluations} />
        </div>

      </div>
    </div>
  );
};
