import React, { useState, useEffect } from 'react';
import { Database } from 'lucide-react';

export const Evidence = () => {
  const [logs, setLogs] = useState<any[]>([]);

  useEffect(() => {
    // In a real application, you would fetch these from the backend's /api/status endpoint
    const fetchLogs = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status');
        if (response.ok) {
           const data = await response.json();
           setLogs(data.history || []);
        }
      } catch (e) {
        console.error('Failed to fetch evidence logs:', e);
      }
    };
    fetchLogs();
  }, []);

  return (
    <div className="min-h-screen bg-background text-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold tracking-wider text-blue-400 mb-2">RESEARCH EVIDENCE</h1>
        <p className="text-gray-400 mb-4">Recorded runtime evidence from TRACEGUARD executions.</p>
        
        <div className="bg-blue-950/30 border border-blue-900 text-blue-300 p-4 rounded mb-8 font-mono text-sm">
           <Database className="inline w-4 h-4 mr-2" />
           These logs represent canonical runtime evidence of the frozen detector deployment. They are illustrative and are not new benchmark metrics.
        </div>

        {logs.length === 0 ? (
          <div className="glass-panel p-8 text-center text-gray-500">
             No runs recorded yet in this session. Go to the Live Demo or Attack Lab to run a scenario.
          </div>
        ) : (
          <div className="space-y-6">
             {logs.map((log, idx) => (
                <div key={idx} className="glass-panel p-6 border border-gray-800">
                   <div className="flex justify-between items-center mb-4 pb-2 border-b border-gray-800">
                      <div className="font-bold text-gray-300">Run ID: <span className="font-mono text-blue-400 font-normal">{log.run_id}</span></div>
                      <div className={`px-2 py-1 rounded text-xs font-bold ${
                         log.status === 'blocked' ? 'bg-red-950/50 text-red-400' : 'bg-emerald-950/50 text-emerald-400'
                      }`}>
                         {log.status.toUpperCase()}
                      </div>
                   </div>
                   
                   <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                      <div><span className="text-gray-500">Scenario:</span> {log.scenario}</div>
                      <div><span className="text-gray-500">Trajectory Steps:</span> {log.trajectory_length}</div>
                      <div className="col-span-2"><span className="text-gray-500">User Goal:</span> {log.user_goal}</div>
                   </div>

                   {log.blocked_action && (
                      <div className="mt-4 p-4 bg-red-950/20 border border-red-900/50 rounded">
                         <h3 className="text-red-500 font-bold mb-2">PRE-ACTION GATE ACTIVATED</h3>
                         <div className="grid grid-cols-2 gap-2 text-sm font-mono text-gray-300">
                            <div>Blocked Action: <span className="text-red-400">{log.blocked_action.tool}</span></div>
                            <div>P(HIJACKED): <span className="text-red-400">{log.blocked_action.p_hijacked.toFixed(3)}</span></div>
                            <div>Threshold: {log.blocked_action.threshold.toFixed(2)}</div>
                            <div>Detection Step: {log.blocked_action.detection_step}</div>
                         </div>
                      </div>
                   )}
                </div>
             ))}
          </div>
        )}
      </div>
    </div>
  );
};
