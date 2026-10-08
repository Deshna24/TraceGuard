import React, { useRef, useEffect } from 'react';
import type { EventEnvelope } from '../types';

interface SecurityEventFeedProps {
  events: EventEnvelope[];
}

export const SecurityEventFeed: React.FC<SecurityEventFeedProps> = ({ events }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [events]);

  const formatTime = (isoString: string) => {
    const d = new Date(isoString);
    const h = d.getHours().toString().padStart(2, '0');
    const m = d.getMinutes().toString().padStart(2, '0');
    const s = d.getSeconds().toString().padStart(2, '0');
    const ms = d.getMilliseconds().toString().padStart(3, '0');
    return `${h}:${m}:${s}.${ms}`;
  };

  const getEventStyle = (type: string) => {
    switch (type) {
      case 'ACTION_PROPOSED':
      case 'TOOL_STARTED':
        return 'text-blue-400 border-l-2 border-blue-500 pl-2';
      case 'DETECTOR_EVALUATED':
      case 'THRESHOLD_CROSSED':
      case 'GATE_DECISION':
      case 'ACTION_BLOCKED':
        return 'text-red-400 border-l-2 border-red-500 pl-2 bg-red-950/20';
      case 'INJECTION_OBSERVED':
        return 'text-amber-500 border-l-2 border-amber-500 pl-2 bg-amber-950/20';
      default:
        return 'text-gray-400 border-l-2 border-gray-700 pl-2';
    }
  };

  const formatDescription = (env: EventEnvelope) => {
    switch (env.event_type) {
      case 'RUN_STARTED': return 'Run started';
      case 'USER_TASK_RECEIVED': return 'User task received';
      case 'AGENT_STARTED': return 'Agent started';
      case 'ACTION_PROPOSED': return `ACTION PROPOSED\n${env.payload.tool}`;
      case 'TOOL_STARTED': return `TOOL EXECUTION\nSTARTED: ${env.payload.tool}`;
      case 'TOOL_COMPLETED': return `TOOL COMPLETED\n${env.payload.tool}`;
      case 'OBSERVATION_RECEIVED': return 'OBSERVATION RECEIVED';
      case 'TRAJECTORY_UPDATED': return `TRAJECTORY UPDATED (Step ${env.payload.latest_step?.step || env.payload.trajectory_length})`;
      case 'INJECTION_OBSERVED': return 'UNTRUSTED INJECTION OBSERVED';
      case 'DETECTOR_EVALUATED': return `TRACEGUARD EVALUATED\nP(HIJACKED) = ${env.payload.p_hijacked.toFixed(3)}`;
      case 'THRESHOLD_CROSSED': return `THRESHOLD CROSSED\n${env.payload.p_hijacked.toFixed(3)} >= ${env.payload.threshold.toFixed(3)}`;
      case 'GATE_DECISION': return `PRE-ACTION GATE\n${env.payload.decision}`;
      case 'ACTION_BLOCKED': return `TOOL EXECUTION\nNOT INVOKED (BLOCKED)`;
      case 'RUN_COMPLETED': return `RUN COMPLETED (${env.payload.status})`;
      case 'RUNTIME_ERROR': return 'RUNTIME ERROR';
      default: return env.event_type;
    }
  };

  return (
    <div className="glass-panel p-4 flex flex-col gap-2 flex-1 max-h-[400px] bg-gray-900 border border-gray-800 rounded">
      <h2 className="text-sm font-bold border-b border-gray-800 pb-2 text-gray-400">
        SECURITY TIMELINE
      </h2>
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto space-y-4 text-xs font-mono pr-2 scrollbar-thin scrollbar-thumb-gray-700"
      >
        {events.map((env, idx) => {
          return (
            <div key={env.event_id || idx} className={`flex gap-3 py-1 ${getEventStyle(env.event_type)}`}>
              <span className="text-gray-500 whitespace-nowrap">{formatTime(env.timestamp)}</span>
              <span className="whitespace-pre-wrap">{formatDescription(env)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
