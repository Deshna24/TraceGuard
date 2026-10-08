import React, { useState, useEffect } from 'react';
import { Database, Lock } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTraceGuard } from '../hooks/useTraceGuard';

export const ProtectedDatabase: React.FC = () => {
  const { state } = useTraceGuard();
  const [records, setRecords] = useState<any[]>([]);

  useEffect(() => {
    const fetchDatabase = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/database');
        if (response.ok) {
          const data = await response.json();
          setRecords(data.records || []);
        } else {
          // Fallback if backend wasn't restarted
          setRecords([
            { customer_id: 'customer_001', name: 'Fictional Ada', status: 'active', order_count: 3 },
            { customer_id: 'customer_002', name: 'Fictional Lin', status: 'inactive', order_count: 1 }
          ]);
        }
      } catch (e) {
        setRecords([
          { customer_id: 'customer_001', name: 'Fictional Ada', status: 'active', order_count: 3 },
          { customer_id: 'customer_002', name: 'Fictional Lin', status: 'inactive', order_count: 1 }
        ]);
      }
    };
    fetchDatabase();
  }, []);

  const isDatabaseProposed = state.currentTool?.includes('database');
  const isHijacked = state.status === 'BLOCKED' && state.blockedActionReason;
  const dbExecutions = state.toolExecutionCounts['database'] || 0;

  if (!isDatabaseProposed && !isHijacked) {
    return null; // Only show when relevant to the attack or when requested
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-6 border-2 border-indigo-900/50 bg-indigo-950/10 rounded flex flex-col gap-4 mt-4 relative overflow-hidden">
      
      {/* Label indicating this is a protected resource */}
      <div className="absolute top-0 right-0 bg-indigo-900/50 text-indigo-300 text-[10px] uppercase font-bold tracking-widest px-3 py-1 rounded-bl-lg">
        Protected Resource
      </div>

      <h2 className="text-xl font-bold flex items-center gap-2 text-indigo-400 uppercase tracking-wider mb-2">
        <Database className="w-6 h-6" /> Controlled Database
      </h2>

      {/* Database State Visualization */}
      <div className="bg-gray-950 border border-gray-800 rounded p-4 font-mono text-sm overflow-x-auto relative">
        <div className="absolute inset-0 bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(79,70,229,0.05)_10px,rgba(79,70,229,0.05)_20px)] pointer-events-none"></div>
        <table className="w-full text-left text-gray-300 border-collapse">
          <thead>
            <tr className="border-b border-gray-800 text-gray-500">
              <th className="py-2 font-normal uppercase">ID</th>
              <th className="py-2 font-normal uppercase">Name</th>
              <th className="py-2 font-normal uppercase">Status</th>
              <th className="py-2 font-normal uppercase">Orders</th>
            </tr>
          </thead>
          <tbody>
            {records.map((rec, idx) => (
              <tr key={idx} className="border-b border-gray-800/50 last:border-0 hover:bg-gray-900/50 transition-colors">
                <td className="py-2 text-indigo-300">{rec.customer_id}</td>
                <td className="py-2">{rec.name}</td>
                <td className="py-2">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${rec.status === 'active' ? 'bg-emerald-900/30 text-emerald-400' : 'bg-gray-800 text-gray-400'}`}>
                    {rec.status.toUpperCase()}
                  </span>
                </td>
                <td className="py-2">{rec.order_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
        
        {/* Fake watermark to clarify it's mock data */}
        <div className="absolute bottom-1 right-2 text-[10px] text-gray-600 font-bold tracking-widest uppercase">
          FICTIONAL / SANDBOXED DATA
        </div>
      </div>

      {/* Database Execution Monitor / What would have happened */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
        
        {/* Scenario Analysis */}
        <div className="bg-gray-950 p-4 border border-gray-800 rounded font-mono text-xs">
          <div className="text-gray-500 font-bold mb-3 border-b border-gray-800 pb-2 uppercase tracking-wider">Target Analysis</div>
          
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Proposed World:</span>
              <span className="text-amber-400 text-right">Read / Export Records</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Target Resource:</span>
              <span className="text-indigo-400 text-right">Controlled Database</span>
            </div>
            
            <div className="my-1 border-t border-gray-800"></div>
            
            <div className="flex justify-between font-bold">
              <span className="text-gray-400">Actual World:</span>
              <span className={isHijacked ? "text-emerald-400" : "text-gray-400"}>
                {isHijacked ? "UNCHANGED / PROTECTED" : "WAITING"}
              </span>
            </div>
            <div className="flex justify-between font-bold">
              <span className="text-gray-400">Records Exported:</span>
              <span className={isHijacked ? "text-emerald-400" : "text-gray-400"}>0</span>
            </div>
          </div>
        </div>

        {/* Execution Monitor */}
        <div className="bg-gray-950 p-4 border border-gray-800 rounded font-mono text-xs flex flex-col">
          <div className="text-gray-500 font-bold mb-3 border-b border-gray-800 pb-2 uppercase tracking-wider">Database Activity</div>
          
          <div className="grid grid-cols-2 gap-y-2 mb-auto text-gray-300">
            <div>Reads:</div><div className="text-right">0</div>
            <div>Writes:</div><div className="text-right">0</div>
            <div>Exports:</div><div className="text-right">{dbExecutions}</div>
          </div>

          {isHijacked && (
            <div className="mt-4 p-2 bg-emerald-950/20 border border-emerald-900/50 rounded flex items-center gap-2 text-emerald-400 font-bold text-[10px] uppercase tracking-wider">
              <Lock className="w-4 h-4" /> Export Prevented
            </div>
          )}
        </div>
      </div>
      
    </motion.div>
  );
};
