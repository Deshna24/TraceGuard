import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Network } from 'lucide-react';
import type { TrajectoryStep } from '../types';

interface LiveTrajectoryProps {
  trajectory: TrajectoryStep[];
}

export const LiveTrajectory: React.FC<LiveTrajectoryProps> = ({ trajectory }) => {
  return (
    <div className="glass-panel p-6 flex flex-col gap-4 flex-1 overflow-hidden" style={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '0.5rem', borderStyle: 'solid', borderWidth: '1px' }}>
      <h2 className="text-xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2">
        <Network className="w-5 h-5 text-blue-400" />
        LIVE TRAJECTORY
      </h2>
      <div className="flex-1 overflow-y-auto pr-2 space-y-4">
        <AnimatePresence>
          {trajectory.map((step, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex gap-4 relative"
            >
              <div className="flex flex-col items-center">
                <div className="w-8 h-8 rounded-full bg-blue-900 flex items-center justify-center text-sm font-bold text-blue-300 border border-blue-700 z-10">
                  {step.step}
                </div>
                {idx < trajectory.length - 1 && (
                  <div className="w-0.5 bg-gray-700 h-full absolute top-8 left-4"></div>
                )}
              </div>
              <div className="flex-1 bg-gray-900/50 p-4 rounded border border-gray-700 text-sm mb-4">
                <div className="font-bold text-gray-300 mb-2">AGENT ACTION</div>
                <div className="font-mono text-blue-400 mb-2">{step.tool}({JSON.stringify(step.tool_input)})</div>
                
                <div className="font-bold text-gray-300 mb-1 mt-4">TOOL OBSERVATION</div>
                <div className="text-gray-400 p-2 bg-gray-950 rounded border border-gray-800 break-words whitespace-pre-wrap">
                   {typeof step.tool_observation === 'string' 
                      ? step.tool_observation 
                      : JSON.stringify(step.tool_observation, null, 2)}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
};
