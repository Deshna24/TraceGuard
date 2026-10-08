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

  const getEventIconAndColor = (type: string) => {
    switch (type) {
      case 'RUN_STARTED':
      case 'USER_TASK_RECEIVED':
      case 'AGENT_STARTED':
      case 'TOOL_COMPLETED':
      case 'OBSERVATION_RECEIVED':
      case 'TRAJECTORY_UPDATED':
      case 'RUN_COMPLETED':
        return { icon: '✓', color: 'text-emerald-400' };
      case 'ACTION_PROPOSED':
      case 'TOOL_STARTED':
        return { icon: '•', color: 'text-blue-400' };
      case 'INJECTION_OBSERVED':
        return { icon: '⚠', color: 'text-amber-500' };
      case 'THRESHOLD_CROSSED':
      case 'ACTION_BLOCKED':
        return { icon: '🚨', color: 'text-red-500' };
      case 'GATE_DECISION':
        return { icon: '🛡', color: 'text-gray-400' };
      default:
        return { icon: 'ℹ', color: 'text-gray-500' };
    }
  };

  const formatDescription = (env: EventEnvelope) => {
    switch (env.event_type) {
      case 'RUN_STARTED': return 'Run started';
      case 'USER_TASK_RECEIVED': return 'User task received';
      case 'AGENT_STARTED': return 'Agent started';
      case 'ACTION_PROPOSED': return `Action proposed: ${env.payload.tool}`;
      case 'TOOL_STARTED': return `Tool started: ${env.payload.tool}`;
      case 'TOOL_COMPLETED': return `Tool completed: ${env.payload.tool}`;
      case 'OBSERVATION_RECEIVED': return 'Observation received';
      case 'TRAJECTORY_UPDATED': return `Trajectory updated (Step ${env.payload.latest_step?.step || env.payload.trajectory_length})`;
      case 'INJECTION_OBSERVED': return 'Untrusted injection observed';
      case 'DETECTOR_EVALUATED': return `Evaluated P(HIJACKED)=${env.payload.p_hijacked.toFixed(2)}`;
      case 'THRESHOLD_CROSSED': return `Threshold crossed (P=${env.payload.p_hijacked.toFixed(2)})`;
      case 'GATE_DECISION': return `Gate decision: ${env.payload.decision}`;
      case 'ACTION_BLOCKED': return 'Action blocked';
      case 'RUN_COMPLETED': return `Run completed (${env.payload.status})`;
      case 'RUNTIME_ERROR': return 'Runtime error';
      default: return env.event_type;
    }
  };

  return (
    <div className="glass-panel p-4 flex flex-col gap-2 flex-1 max-h-[300px] bg-gray-900 border border-gray-800 rounded">
      <h2 className="text-sm font-bold border-b border-gray-800 pb-2 text-gray-400">
        SECURITY EVENTS
      </h2>
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto space-y-2 text-sm font-mono pr-2 scrollbar-thin scrollbar-thumb-gray-700"
      >
        {events.map((env, idx) => {
          const { icon, color } = getEventIconAndColor(env.event_type);
          const time = new Date(env.timestamp).toLocaleTimeString([], { hour12: false });
          return (
            <div key={env.event_id || idx} className="flex gap-2">
              <span className="text-gray-600">[{time}]</span>
              <span className={color}>{icon}</span>
              <span className="text-gray-300">{formatDescription(env)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
