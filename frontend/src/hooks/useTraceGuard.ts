import { useState, useEffect, useCallback, useRef } from 'react';
import { TraceGuardEvent, GateDecision, TrajectoryStep, DetectorEvaluation } from '../types';

export function useTraceGuard() {
  const [status, setStatus] = useState<string>('IDLE');
  const [scenario, setScenario] = useState<string>('');
  const [goal, setGoal] = useState<string>('');
  const [events, setEvents] = useState<TraceGuardEvent[]>([]);
  const [trajectory, setTrajectory] = useState<TrajectoryStep[]>([]);
  const [evaluations, setEvaluations] = useState<DetectorEvaluation[]>([]);
  const [gateDecisions, setGateDecisions] = useState<GateDecision[]>([]);
  const [blockedAction, setBlockedAction] = useState<any | null>(null);
  const [injectionObserved, setInjectionObserved] = useState<boolean>(false);
  const [toolExecutionCount, setToolExecutionCount] = useState<Record<string, number>>({});
  const [currentTool, setCurrentTool] = useState<string | null>(null);
  const [currentAction, setCurrentAction] = useState<any | null>(null);

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    
    ws.current.onmessage = (event) => {
      const data: TraceGuardEvent = JSON.parse(event.data);
      setEvents((prev) => [...prev, data]);
      
      switch (data.event_type) {
        case 'RUN_STARTED':
          setStatus('RUNNING');
          setScenario(data.data.scenario);
          setGoal(data.data.goal);
          break;
        case 'AGENT_STARTED':
          setStatus('THINKING');
          break;
        case 'ACTION_PROPOSED':
          setStatus('ACTING');
          setCurrentAction(data.data);
          break;
        case 'DETECTOR_EVALUATED':
          setEvaluations((prev) => [...prev, data.data]);
          break;
        case 'GATE_DECISION':
          setGateDecisions((prev) => [...prev, data.data]);
          break;
        case 'ACTION_BLOCKED':
          setStatus('BLOCKED');
          setBlockedAction(data.data.action);
          break;
        case 'TOOL_STARTED':
          setCurrentTool(data.data.tool);
          setStatus('WAITING');
          setToolExecutionCount(prev => ({
             ...prev, 
             [data.data.tool]: (prev[data.data.tool] || 0) + 1 
          }));
          break;
        case 'TOOL_COMPLETED':
          setCurrentTool(null);
          setCurrentAction(null);
          setStatus('THINKING');
          break;
        case 'TRAJECTORY_UPDATED':
          setTrajectory((prev) => [...prev, data.data.step]);
          break;
        case 'INJECTION_OBSERVED':
          setInjectionObserved(true);
          break;
        case 'RUN_COMPLETED':
          if (data.data.status === 'blocked') {
             setStatus('BLOCKED');
          } else {
             setStatus('COMPLETED');
          }
          break;
      }
    };

    return () => {
      ws.current?.close();
    };
  }, []);

  const startRun = async (selectedScenario: string) => {
    // Reset state
    setEvents([]);
    setTrajectory([]);
    setEvaluations([]);
    setGateDecisions([]);
    setBlockedAction(null);
    setInjectionObserved(false);
    setToolExecutionCount({});
    setCurrentTool(null);
    setCurrentAction(null);
    setStatus('STARTING');
    
    await fetch('http://localhost:8000/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: selectedScenario })
    });
  };
  
  const reset = () => {
    setEvents([]);
    setTrajectory([]);
    setEvaluations([]);
    setGateDecisions([]);
    setBlockedAction(null);
    setInjectionObserved(false);
    setToolExecutionCount({});
    setCurrentTool(null);
    setCurrentAction(null);
    setStatus('IDLE');
    setScenario('');
    setGoal('');
  };

  return {
    status,
    scenario,
    goal,
    events,
    trajectory,
    evaluations,
    gateDecisions,
    blockedAction,
    injectionObserved,
    toolExecutionCount,
    currentTool,
    currentAction,
    startRun,
    reset
  };
}
