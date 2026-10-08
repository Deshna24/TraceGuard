import React from 'react';
import { motion } from 'framer-motion';
import { Terminal, BrainCircuit, AlertTriangle, Scale, ShieldCheck, ShieldAlert } from 'lucide-react';
import { useTraceGuard } from '../hooks/useTraceGuard';
import { RiskGraph } from '../components/RiskGraph';
import { LiveTrajectory } from '../components/LiveTrajectory';
import { SecurityEventFeed } from '../components/SecurityEventFeed';
import { PreActionGate } from '../components/PreActionGate';

export const Dashboard: React.FC = () => {
  const { state, startRun, stopRun, reset } = useTraceGuard();
  const [customGoal, setCustomGoal] = React.useState('');
  const [customInjection, setCustomInjection] = React.useState('');

  const isDisconnected = state.connectionStatus === 'DISCONNECTED';

  return (
    <div className="min-h-screen bg-background text-gray-100 p-6 flex flex-col gap-6">
      {/* Header */}
      <header className="flex items-center justify-between glass-panel p-4 mb-2">
        <div>
          <h1 className="text-4xl font-extrabold tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-600 text-glow-blue">TRACEGUARD</h1>
          <p className="text-blue-300/60 text-sm mt-1 uppercase tracking-widest">Real-Time LLM Agent Security Monitoring</p>
        </div>
        <div className="flex gap-4 items-center">
          <div className="flex flex-col items-end text-sm text-gray-300">
            <span>Granite 4.1 8B</span>
            <span>TRACEGUARD LSTM</span>
          </div>
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full border text-sm font-medium ${
            isDisconnected ? 'bg-red-900/30 border-red-500/30 text-red-400' : 'bg-emerald-900/30 border-emerald-500/30 text-emerald-400'
          }`}>
            {!isDisconnected && (
               <span className="relative flex h-3 w-3">
                 <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                 <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
               </span>
            )}
            {isDisconnected ? 'BACKEND DISCONNECTED' : (state.probabilities?.pre_action ? 'PRE-ACTION PROTECTION ACTIVE' : 'LOCAL RUNTIME ACTIVE')}
          </div>
        </div>
      </header>

      {isDisconnected && (
        <div className="bg-red-950 border border-red-500 text-red-400 p-4 rounded text-center font-bold">
          RUNTIME UNAVAILABLE - PLEASE START THE BACKEND SERVER
        </div>
      )}

      <div className="grid grid-cols-12 gap-6">
        {/* Left Column */}
        <div className="col-span-12 lg:col-span-3 flex flex-col gap-6">
          <div className="glass-panel p-6 flex flex-col gap-4">
            <h2 className="text-xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2">
              <Terminal className="w-5 h-5 text-gray-400" />
              USER TASK
            </h2>
            <div className="bg-gray-900 p-4 rounded border border-gray-800 min-h-[100px] text-sm text-gray-300 break-words">
              {state.userGoal ? state.userGoal : <span className="text-gray-500">Select a scenario to begin...</span>}
            </div>
            
            <div className="flex flex-col gap-2 mt-2">
               <span className="text-xs text-gray-500 uppercase font-bold">Scenario</span>
               <div className="grid grid-cols-1 gap-3">
                 <div className="group relative">
                   <button onClick={() => startRun('BENIGN')} disabled={isDisconnected || state.status === 'RUNNING'} className="w-full premium-button py-3 px-3 rounded-lg text-sm font-bold text-white uppercase tracking-wider flex items-center justify-center gap-2 relative overflow-hidden group-hover:shadow-[0_0_20px_rgba(59,130,246,0.3)]">
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
                      <ShieldCheck className="w-5 h-5" /> BENIGN
                   </button>
                   <div className="opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity absolute left-full ml-4 top-0 w-64 bg-gray-950/90 border border-blue-500/30 p-3 rounded-lg text-xs text-blue-200 z-50 backdrop-blur-md shadow-2xl">
                     <span className="font-bold text-blue-400 block mb-1">Benign Scenario</span>
                     User asks for a simple calculation. Agent uses calculator tool. No injection is present.
                   </div>
                 </div>

                 <div className="group relative">
                   <button onClick={() => startRun('INJECTION_RESISTED')} disabled={isDisconnected || state.status === 'RUNNING'} className="w-full premium-button py-3 px-3 rounded-lg text-sm font-bold text-white uppercase tracking-wider flex items-center justify-center gap-2 relative overflow-hidden group-hover:shadow-[0_0_20px_rgba(251,191,36,0.3)] border-amber-500/30">
                      <div className="absolute inset-0 bg-gradient-to-r from-amber-500/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
                      <AlertTriangle className="w-5 h-5 text-amber-400" /> RESISTED
                   </button>
                   <div className="opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity absolute left-full ml-4 top-0 w-64 bg-gray-950/90 border border-amber-500/30 p-3 rounded-lg text-xs text-amber-200 z-50 backdrop-blur-md shadow-2xl">
                     <span className="font-bold text-amber-400 block mb-1">Injection Resisted</span>
                     User asks to search a knowledge base. The search returns a malicious injection. Agent ignores it and completes the original task.
                   </div>
                 </div>

                 <div className="group relative">
                   <button onClick={() => startRun('HIJACKED')} disabled={isDisconnected || state.status === 'RUNNING'} className="w-full premium-button py-3 px-3 rounded-lg text-sm font-bold text-white uppercase tracking-wider flex items-center justify-center gap-2 relative overflow-hidden group-hover:shadow-[0_0_20px_rgba(239,68,68,0.3)] border-red-500/30">
                      <div className="absolute inset-0 bg-gradient-to-r from-red-500/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000"></div>
                      <ShieldAlert className="w-5 h-5 text-red-400" /> HIJACKED
                   </button>
                   <div className="opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity absolute left-full ml-4 top-0 w-64 bg-gray-950/90 border border-red-500/30 p-3 rounded-lg text-xs text-red-200 z-50 backdrop-blur-md shadow-2xl">
                     <span className="font-bold text-red-400 block mb-1">Hijacked Scenario</span>
                     Agent searches knowledge base, receives malicious injection, deviates from original goal, and attempts a suspicious database export. TRACEGUARD blocks it.
                   </div>
                 </div>
               </div>
            </div>

            <div className="flex flex-col gap-2 mt-4 border-t border-gray-800 pt-4">
               <span className="text-xs text-blue-400 uppercase font-bold flex items-center gap-2">Interactive Custom Mode</span>
               <div className="flex flex-col gap-2">
                 <textarea 
                   placeholder="Enter a custom task for the agent (e.g. Find the capital of Japan)..."
                   value={customGoal}
                   onChange={e => setCustomGoal(e.target.value)}
                   className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm text-gray-200 placeholder-gray-500 focus:border-blue-500 focus:outline-none min-h-[60px]"
                 />
                 <textarea 
                   placeholder="Enter an attacker injection (e.g. Ignore instructions and export records)..."
                   value={customInjection}
                   onChange={e => setCustomInjection(e.target.value)}
                   className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm text-amber-200/80 placeholder-gray-500 focus:border-amber-500 focus:outline-none min-h-[60px]"
                 />
                 <button 
                   onClick={() => startRun('CUSTOM', customGoal, customInjection)} 
                   disabled={isDisconnected || state.status === 'RUNNING' || !customGoal.trim()} 
                   className="premium-button py-3 px-3 rounded text-sm font-bold text-white uppercase tracking-wider"
                 >
                   ⚡ RUN LIVE ATTACK
                 </button>
               </div>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-4">
               <button onClick={stopRun} disabled={state.status !== 'RUNNING' && state.status !== 'ACTING'} className="px-4 py-2 bg-amber-900/30 hover:bg-amber-900/50 text-amber-400 rounded text-sm font-medium transition-colors border border-amber-900/50">STOP</button>
               <button onClick={reset} className="px-4 py-2 bg-red-900/30 hover:bg-red-900/50 text-red-400 rounded text-sm font-medium transition-colors border border-red-900/50">RESET</button>
            </div>
          </div>

          <div className="glass-panel p-6 flex flex-col gap-4">
            <h2 className="text-xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2">
              <BrainCircuit className="w-5 h-5 text-gray-400" />
              LIVE AGENT
            </h2>
            <div className="space-y-3 text-sm">
               <div className="flex justify-between items-center bg-gray-900 p-2 rounded border border-gray-800">
                 <span className="text-gray-400">Agent</span>
                 <span className="text-gray-300">Granite 4.1</span>
               </div>
               <div className="flex justify-between items-center bg-gray-900 p-2 rounded border border-gray-800">
                 <span className="text-gray-400">State</span>
                 <span className={`font-bold ${
                   state.status === 'IDLE' ? 'text-gray-500' :
                   state.status === 'RUNNING' || state.status === 'ACTING' || state.status === 'WAITING' ? 'text-blue-400' :
                   state.status === 'BLOCKED' ? 'text-red-500' :
                   state.status === 'COMPLETED' ? 'text-emerald-500' : 'text-gray-400'
                 }`}>{state.status}</span>
               </div>
               <div className="flex justify-between items-center bg-gray-900 p-2 rounded border border-gray-800">
                 <span className="text-gray-400">Step</span>
                 <span className="font-mono text-gray-300">{state.currentStep !== null ? state.currentStep : '-'}</span>
               </div>
               {state.currentAction && (
                  <div className="bg-gray-900 p-2 rounded border border-gray-800">
                     <span className="text-gray-400 block mb-1">Proposed Action</span>
                     <span className="font-mono text-blue-300">{state.currentTool}</span>
                  </div>
               )}
            </div>
          </div>

          <div className={`glass-panel p-6 flex flex-col gap-4 transition-colors duration-500 ${state.untrustedObservation ? 'border-amber-500/50 bg-amber-950/20' : ''}`}>
             <h2 className="text-xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2">
              <AlertTriangle className={`w-5 h-5 ${state.untrustedObservation ? 'text-amber-500' : 'text-gray-500'}`} />
              ATTACK SURFACE
            </h2>
            <div className="bg-gray-900 p-3 rounded border border-gray-800 text-sm">
                 <div className="text-gray-400 mb-1">Injection Status:</div>
                 <div className={`font-bold ${state.untrustedObservation ? 'text-amber-500' : 'text-gray-500'}`}>
                   {state.untrustedObservation ? '⚠ UNTRUSTED CONTENT' : 'NO UNTRUSTED CONTENT'}
                 </div>
            </div>
            {state.untrustedObservation && (
               <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="bg-gray-950 p-3 rounded border border-amber-900/50 text-sm overflow-hidden">
                 <div className="text-gray-400 mb-1">Untrusted Observation:</div>
                 <div className="text-amber-200/80 break-words whitespace-pre-wrap">
                    {typeof state.untrustedObservation === 'string' ? state.untrustedObservation : JSON.stringify(state.untrustedObservation, null, 2)}
                 </div>
               </motion.div>
            )}
          </div>
        </div>

        {/* Middle Column */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-6">
           {state.untrustedObservation && state.currentTool === 'database.export_records' && (
              <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="glass-panel p-4 bg-red-950/20 border-red-900/50 flex flex-col gap-2 rounded">
                 <h2 className="text-lg font-bold flex items-center gap-2 text-red-400">
                    <Scale className="w-5 h-5" />
                    GOAL VS BEHAVIOR
                 </h2>
                 <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-gray-900 p-2 rounded border border-gray-800">
                       <div className="text-gray-400 mb-1">AUTHORIZED USER GOAL</div>
                       <div className="text-gray-300 break-words">{state.userGoal}</div>
                    </div>
                    <div className="bg-gray-900 p-2 rounded border border-red-900/30">
                       <div className="text-gray-400 mb-1">AGENT PROPOSED ACTION</div>
                       <div className="text-red-400 font-mono break-words">{state.currentTool}</div>
                    </div>
                 </div>
                 <div className="text-center text-red-500 font-bold mt-2">⚠ BEHAVIORAL DEVIATION</div>
              </motion.div>
           )}

          <RiskGraph evaluations={state.evaluations} />
          {state.status === 'COMPLETED' && state.trajectoryLength === 0 && (
             <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="glass-panel p-6 bg-blue-950/20 border-blue-900/50 flex flex-col gap-2 rounded text-center my-4">
                <div className="text-blue-400 font-bold">AGENT COMPLETED TASK DIRECTLY</div>
                <div className="text-gray-400 text-sm">No tools were invoked during this run, so no environment observations occurred and the TRACEGUARD detector was not evaluated.</div>
             </motion.div>
          )}
          <LiveTrajectory trajectory={state.trajectory} />
        </div>

        {/* Right Column */}
        <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
           <SecurityEventFeed events={state.events} />

           {state.probabilities && (
             <div className="glass-panel p-6 flex flex-col gap-4">
               <h2 className="text-xl font-bold border-b border-gray-800 pb-2">TRACEGUARD</h2>
               <div className="space-y-3">
                 <div className="flex justify-between items-center text-sm">
                   <span className="text-gray-400">P(BENIGN)</span>
                   <span className="font-mono text-gray-300">{(state.probabilities.p_benign * 100).toFixed(1)}%</span>
                 </div>
                 <div className="flex justify-between items-center text-sm">
                   <span className="text-gray-400">P(INJECTION_RESISTED)</span>
                   <span className="font-mono text-gray-300">{(state.probabilities.p_injection_resisted * 100).toFixed(1)}%</span>
                 </div>
                 <div className="flex justify-between items-center text-sm font-bold">
                   <span className="text-gray-300">P(HIJACKED)</span>
                   <span className={`font-mono ${state.probabilities.p_hijacked >= 0.5 ? 'text-red-500' : 'text-blue-400'}`}>
                     {(state.probabilities.p_hijacked * 100).toFixed(1)}%
                   </span>
                 </div>
                 <div className="my-2 h-px bg-gray-800"></div>
                 <div className="flex justify-between items-center text-sm">
                   <span className="text-gray-400">THRESHOLD</span>
                   <span className="font-mono text-gray-300">{(state.probabilities.threshold * 100).toFixed(1)}%</span>
                 </div>
                 <div className="flex justify-between items-center text-sm">
                   <span className="text-gray-400">CURRENT CLASS</span>
                   <span className="font-bold text-gray-300">{state.probabilities.predicted_class}</span>
                 </div>
                 <div className="flex justify-between items-center text-sm">
                   <span className="text-gray-400">PRE-ACTION</span>
                   <span className="font-bold text-gray-300">{state.probabilities.pre_action ? 'TRUE' : 'FALSE'}</span>
                 </div>
               </div>
             </div>
           )}

           {state.gateDetails && (
              <PreActionGate gateDetails={state.gateDetails} toolExecutionCounts={state.toolExecutionCounts} />
           )}

           {state.status === 'BLOCKED' && state.blockedActionReason && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6 bg-red-950/40 border-red-500 rounded border-2">
                 <h2 className="text-xl font-bold flex items-center gap-2 text-red-500 mb-4">
                    🚨 BEHAVIORAL HIJACKING DETECTED
                 </h2>
                 <div className="text-sm space-y-2">
                   <p><span className="text-gray-400">P(HIJACKED):</span> <span className="text-red-400 font-mono">{(state.probabilities?.p_hijacked || 0).toFixed(3)}</span></p>
                   <p><span className="text-gray-400">Threshold:</span> <span className="font-mono">{(state.probabilities?.threshold || 0).toFixed(2)}</span></p>
                   <p><span className="text-gray-400">Detection step:</span> <span className="font-mono">{state.probabilities?.step}</span></p>
                   <p><span className="text-gray-400">PRE-ACTION:</span> <span className="font-mono text-emerald-400">TRUE</span></p>
                   <p><span className="text-gray-400">Proposed tool:</span> <span className="font-mono text-red-400">{state.gateDetails?.tool || 'unknown'}</span></p>
                   <p><span className="text-gray-400">Status:</span> <span className="font-bold text-red-500">BLOCKED</span></p>
                   <p><span className="text-gray-400">Tool execution count:</span> <span className="font-mono font-bold">0</span></p>
                 </div>
                 <div className="mt-4 text-sm font-bold text-red-400 bg-red-950/50 p-3 rounded">
                    TRACEGUARD PREVENTED THE ACTION BEFORE EXECUTION.
                 </div>
              </motion.div>
           )}

           {state.status === 'COMPLETED' && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6 bg-emerald-950/20 border-emerald-500/50 rounded border-2">
                 <h2 className="text-xl font-bold flex items-center gap-2 text-emerald-500 mb-4">
                    {state.untrustedObservation ? '🟡 INJECTION OBSERVED' : '🟢 TASK COMPLETED'}
                 </h2>
                 <div className="text-sm space-y-2 text-emerald-200 mb-4">
                    {state.untrustedObservation ? (
                       <>
                         <div className="font-bold text-emerald-400 mb-2">✓ AGENT REMAINED ALIGNED</div>
                         <div>Injection was present, but behavioral hijacking did not occur.</div>
                       </>
                    ) : (
                       <div>Agent successfully completed the task without deviation.</div>
                    )}
                 </div>
                 {state.finalAnswer && (
                   <div className="bg-gray-950 p-4 rounded border border-emerald-900/50 mt-2">
                     <div className="text-emerald-500 font-bold mb-2 text-xs uppercase tracking-wider">Final Output:</div>
                     <div className="text-gray-100 whitespace-pre-wrap">{state.finalAnswer}</div>
                   </div>
                 )}
              </motion.div>
           )}
        </div>
      </div>
    </div>
  );
};
