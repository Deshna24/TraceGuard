import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Network, Search, Database, Calculator, AlertTriangle, ShieldCheck } from 'lucide-react';
import type { TrajectoryStep } from '../types';

interface LiveTrajectoryProps {
  trajectory: TrajectoryStep[];
  currentTool: string | null;
  status: string;
}

export const LiveTrajectory: React.FC<LiveTrajectoryProps> = ({ trajectory, currentTool, status }) => {
  const getToolIcon = (toolName: string | undefined | null) => {
    if (!toolName) return <Network className="w-5 h-5" />;
    if (toolName.includes('search')) return <Search className="w-5 h-5" />;
    if (toolName.includes('database')) return <Database className="w-5 h-5" />;
    if (toolName.includes('calculator')) return <Calculator className="w-5 h-5" />;
    return <Network className="w-5 h-5" />;
  };

  const isUntrusted = (obs: any) => {
    if (typeof obs === 'string') {
      return obs.includes('IMPORTANT:') || obs.includes('IGNORE');
    }
    return false;
  };

  return (
    <div className="glass-panel p-6 flex flex-col gap-4 flex-1 overflow-hidden" style={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '0.5rem', borderStyle: 'solid', borderWidth: '1px' }}>
      <h2 className="text-xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2 text-gray-300">
        <Network className="w-5 h-5 text-blue-400" />
        AGENT TRAJECTORY
      </h2>
      <div className="flex-1 overflow-y-auto pr-2 space-y-6 scrollbar-thin scrollbar-thumb-gray-700">
        <AnimatePresence>
          {trajectory.map((step, idx) => {
            const hasUntrusted = isUntrusted(step.tool_observation);
            return (
              <motion.div 
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex gap-4 relative"
              >
                <div className="flex flex-col items-center mt-1">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold z-10 shadow-lg ${
                    hasUntrusted ? 'bg-amber-900 border border-amber-500 text-amber-200' : 'bg-blue-900 border border-blue-500 text-blue-200'
                  }`}>
                    {step.step}
                  </div>
                  {idx < trajectory.length - 1 && (
                    <div className="w-0.5 bg-gray-700 h-full absolute top-9 left-4"></div>
                  )}
                  {idx === trajectory.length - 1 && currentTool && (
                     <div className="w-0.5 bg-blue-500/30 h-full absolute top-9 left-4 animate-pulse"></div>
                  )}
                </div>
                
                <div className={`flex-1 p-4 rounded border text-sm mb-2 shadow-lg ${
                  hasUntrusted ? 'bg-amber-950/20 border-amber-900/50' : 'bg-gray-900/50 border-gray-700'
                }`}>
                  <div className="flex justify-between items-center mb-3 border-b border-gray-800 pb-2">
                    <div className="font-bold flex items-center gap-2 text-gray-200">
                      {getToolIcon(step.tool)}
                      STEP {step.step} — {step.tool.toUpperCase()}
                    </div>
                    <div className="text-xs bg-emerald-950/50 text-emerald-400 px-2 py-1 rounded border border-emerald-900/50 font-bold flex items-center gap-1">
                      <ShieldCheck className="w-3 h-3" /> EXECUTED
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 gap-3">
                    <div>
                      <div className="text-xs text-gray-500 mb-1 font-bold">AGENT INPUT</div>
                      <div className="font-mono text-blue-300 bg-gray-950 p-2 rounded">
                        {JSON.stringify(step.tool_input)}
                      </div>
                    </div>
                    
                    <div>
                      <div className={`text-xs mb-1 font-bold flex items-center gap-1 ${hasUntrusted ? 'text-amber-500' : 'text-gray-500'}`}>
                        {hasUntrusted && <AlertTriangle className="w-3 h-3" />}
                        {hasUntrusted ? 'UNTRUSTED OBSERVATION' : 'TOOL OBSERVATION'}
                      </div>
                      <div className={`p-2 rounded font-mono text-xs break-words whitespace-pre-wrap ${
                        hasUntrusted ? 'bg-amber-950/30 text-amber-200 border border-amber-900/50' : 'bg-gray-950 text-gray-400'
                      }`}>
                         {typeof step.tool_observation === 'string' 
                            ? step.tool_observation 
                            : JSON.stringify(step.tool_observation, null, 2)}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            );
          })}
          
          {/* Show pending action if acting */}
          {currentTool && status !== 'COMPLETED' && (
            <motion.div 
              key="pending"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-4 relative mt-2"
            >
              <div className="flex flex-col items-center mt-1">
                <div className="w-8 h-8 rounded-full bg-gray-800 flex items-center justify-center text-sm font-bold border border-gray-600 text-gray-400 z-10 animate-pulse">
                  {trajectory.length + 1}
                </div>
              </div>
              
              <div className="flex-1 bg-gray-900/30 p-4 rounded border border-gray-700/50 text-sm mb-2 shadow-lg opacity-70">
                <div className="flex justify-between items-center mb-2 border-b border-gray-800/50 pb-2">
                  <div className="font-bold flex items-center gap-2 text-gray-400">
                    {getToolIcon(currentTool)}
                    STEP {trajectory.length + 1} — {currentTool.toUpperCase()}
                  </div>
                  <div className={`text-xs px-2 py-1 rounded font-bold ${
                    status === 'BLOCKED' ? 'bg-red-950/50 text-red-500 border border-red-900/50' : 'bg-amber-950/50 text-amber-500 border border-amber-900/50 animate-pulse'
                  }`}>
                    {status === 'BLOCKED' ? 'BLOCKED BEFORE EXECUTION' : 'AWAITING SECURITY DECISION'}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
