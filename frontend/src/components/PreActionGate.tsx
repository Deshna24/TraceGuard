import React from 'react';
import { ShieldCheck, ShieldAlert } from 'lucide-react';
import { motion } from 'framer-motion';

interface PreActionGateProps {
  gateDetails: any;
  toolExecutionCounts: Record<string, number>;
}

export const PreActionGate: React.FC<PreActionGateProps> = ({ gateDetails, toolExecutionCounts }) => {
  if (!gateDetails) return null;

  const isBlocked = gateDetails.decision === 'BLOCK';
  const executionCount = toolExecutionCounts[gateDetails.tool] || 0;

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`glass-panel p-6 flex flex-col gap-4 border-2 ${
        isBlocked ? 'bg-red-950/20 border-red-500/50' : 'bg-emerald-950/20 border-emerald-500/50'
      }`}
    >
      <h2 className="text-xl font-bold flex items-center justify-center gap-2">
        {isBlocked ? <ShieldAlert className="w-6 h-6 text-red-500" /> : <ShieldCheck className="w-6 h-6 text-emerald-500" />}
        <span className={isBlocked ? "text-red-500" : "text-emerald-500"}>TRACEGUARD PRE-ACTION GATE</span>
      </h2>

      <div className="grid grid-cols-2 gap-4 text-sm mt-2">
        <div className="bg-gray-900/50 p-3 rounded">
          <div className="text-gray-400 mb-1">Proposed Action</div>
          <div className="font-mono text-blue-300">{gateDetails.tool}</div>
        </div>
        <div className="bg-gray-900/50 p-3 rounded">
          <div className="text-gray-400 mb-1">P(HIJACKED) / Threshold</div>
          <div className="font-mono text-gray-300">
            {gateDetails.p_hijacked.toFixed(3)} / {gateDetails.threshold.toFixed(2)}
          </div>
        </div>
        <div className="bg-gray-900/50 p-3 rounded">
          <div className="text-gray-400 mb-1">Detection Step</div>
          <div className="font-mono text-gray-300">{gateDetails.detection_step}</div>
        </div>
        <div className="bg-gray-900/50 p-3 rounded">
          <div className="text-gray-400 mb-1">Execution Status</div>
          <div className="font-mono text-gray-300">
            {isBlocked ? 'BLOCKED' : 'ALLOWED'}
          </div>
        </div>
      </div>

      <div className="bg-gray-900/50 p-3 rounded flex justify-between items-center text-sm">
         <div className="text-gray-400">Execution Count ({gateDetails.tool})</div>
         <div className="font-mono text-gray-300 font-bold">{executionCount}</div>
      </div>

      {isBlocked && (
        <div className="mt-4 flex flex-col items-center justify-center space-y-2 text-red-400 font-mono text-sm">
          <div>AGENT ACTION</div>
          <div>↓</div>
          <div>TRACEGUARD</div>
          <div>↓</div>
          <div className="font-bold text-red-500 bg-red-950 px-4 py-1 rounded">BLOCK</div>
          <div>↓</div>
          <div>TOOL NOT EXECUTED</div>
        </div>
      )}
    </motion.div>
  );
};
