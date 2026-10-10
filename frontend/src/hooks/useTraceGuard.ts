import { useState, useEffect, useCallback, useRef } from 'react';
import type { EventEnvelope, RunState } from '../types';

export function useTraceGuard() {
  const [state, setState] = useState<RunState>({
    runId: null,
    status: 'IDLE',
    scenario: null,
    userGoal: null,
    model: null,
    currentStep: null,
    currentAction: null,
    currentTool: null,
    events: [],
    trajectory: [],
    trajectoryLength: 0,
    untrustedObservation: null,
    injectionNote: null,
    probabilities: null,
    evaluations: [],
    thresholdCrossed: false,
    gateDecision: null,
    gateDetails: null,
    blockedActionReason: null,
    finalAnswer: null,
    toolExecutionCounts: {},
    error: null,
    connectionStatus: 'DISCONNECTED'
  });

  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<number | null>(null);

  const connect = useCallback(() => {
    setState(prev => ({ ...prev, connectionStatus: 'CONNECTING' }));
    
    // In production, you would probably want to use an env var for the WebSocket URL
    const wsUrl = 'ws://localhost:8000/ws';
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      setState(prev => ({ ...prev, connectionStatus: 'CONNECTED' }));
    };

    ws.current.onclose = () => {
      setState(prev => ({ ...prev, connectionStatus: 'DISCONNECTED' }));
      // Attempt to reconnect after 3 seconds
      reconnectTimeout.current = window.setTimeout(connect, 3000);
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket Error:', error);
      // Close will be called, which handles reconnection
    };

    ws.current.onmessage = (event) => {
      const envelope: EventEnvelope = JSON.parse(event.data);
      
      setState(prev => {
        const nextEvents = [...prev.events, envelope];
        let nextState = { ...prev, events: nextEvents };

        switch (envelope.event_type) {
          case 'RUN_STARTED':
            nextState = {
              ...prev, // Keep connectionStatus
              status: 'RUNNING',
              runId: envelope.run_id,
              scenario: envelope.payload.scenario,
              userGoal: envelope.payload.user_goal,
              model: null,
              currentStep: null,
              currentAction: null,
              currentTool: null,
              events: [envelope], // Clear previous events
              trajectory: [],
              trajectoryLength: 0,
              untrustedObservation: null,
              injectionNote: null,
              probabilities: null,
              evaluations: [],
              thresholdCrossed: false,
              gateDecision: null,
              gateDetails: null,
              blockedActionReason: null,
              finalAnswer: null,
              toolExecutionCounts: {},
              error: null
            };
            break;
            
          case 'USER_TASK_RECEIVED':
            nextState.userGoal = envelope.payload.user_goal;
            break;
            
          case 'AGENT_STARTED':
            nextState.model = envelope.payload.model;
            nextState.status = 'RUNNING'; // or IDLE? RUNNING is good
            break;
            
          case 'ACTION_PROPOSED':
            nextState.currentAction = envelope.payload.action;
            nextState.currentTool = envelope.payload.tool;
            nextState.status = 'ACTING';
            break;
            
          case 'TOOL_STARTED':
            nextState.currentTool = envelope.payload.tool;
            nextState.status = 'WAITING';
            // Optimistically update count
            nextState.toolExecutionCounts = {
               ...nextState.toolExecutionCounts,
               [envelope.payload.tool]: (nextState.toolExecutionCounts[envelope.payload.tool] || 0) + 1
            };
            break;
            
          case 'TOOL_COMPLETED':
            nextState.currentTool = null;
            nextState.currentAction = null;
            nextState.status = 'RUNNING';
            break;
            
          case 'OBSERVATION_RECEIVED':
            // Observation received, but wait for trajectory update to show in list
            break;
            
          case 'TRAJECTORY_UPDATED':
            nextState.trajectoryLength = envelope.payload.trajectory_length;
            if (envelope.payload.latest_step) {
               const newStep = envelope.payload.latest_step;
               const existingIndex = nextState.trajectory.findIndex(s => s.step === newStep.step);
               if (existingIndex >= 0) {
                 const newTrajectory = [...nextState.trajectory];
                 newTrajectory[existingIndex] = newStep;
                 nextState.trajectory = newTrajectory;
               } else {
                 nextState.trajectory = [...nextState.trajectory, newStep];
               }
               nextState.currentStep = newStep.step;
            }
            break;
            
          case 'INJECTION_OBSERVED':
            nextState.untrustedObservation = envelope.payload.observation;
            nextState.injectionNote = envelope.payload.note;
            break;
            
          case 'DETECTOR_EVALUATED':
            nextState.probabilities = envelope.payload;
            nextState.evaluations = [...nextState.evaluations, envelope.payload];
            break;
            
          case 'THRESHOLD_CROSSED':
            nextState.thresholdCrossed = true;
            break;
            
          case 'GATE_DECISION':
            nextState.gateDecision = envelope.payload.decision;
            nextState.gateDetails = envelope.payload;
            break;
            
          case 'ACTION_BLOCKED':
            nextState.status = 'BLOCKED';
            nextState.blockedActionReason = envelope.payload.reason;
            break;
            
          case 'RUN_COMPLETED':
            nextState.status = envelope.payload.status === 'blocked' ? 'BLOCKED' : 'COMPLETED';
            if (envelope.payload.tool_execution_counts) {
              nextState.toolExecutionCounts = envelope.payload.tool_execution_counts;
            }
            if (envelope.payload.answer) {
              nextState.finalAnswer = envelope.payload.answer;
            }
            break;
            
          case 'RUNTIME_ERROR':
            nextState.status = 'ERROR';
            nextState.error = envelope.payload.error;
            break;
        }

        return nextState;
      });
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
      if (ws.current) {
        ws.current.onmessage = null;
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, [connect]);

  const startRun = async (
    scenarioName: string, 
    options?: { customGoal?: string; customInjection?: string; injectionTarget?: string }
  ) => {
    try {
      const endpoint = 'http://localhost:8000/api/run/start/live';
      await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          scenario: scenarioName,
          ...(options?.customGoal && { custom_goal: options.customGoal }),
          ...(options?.customInjection && { custom_injection: options.customInjection }),
          ...(options?.injectionTarget && { injection_target: options.injectionTarget })
        })
      });
    } catch (e) {
      console.error('Failed to start run:', e);
    }
  };

  const stopRun = async () => {
    try {
      await fetch('http://localhost:8000/api/run/stop', { method: 'POST' });
    } catch (e) {
      console.error('Failed to stop run:', e);
    }
  };

  const reset = async () => {
    try {
      await fetch('http://localhost:8000/api/run/reset', { method: 'POST' });
    } catch (e) {
      console.error('Failed to reset run on backend:', e);
    }
    
    // Clear frontend state
    setState(prev => ({
      ...prev,
      runId: null,
      status: 'IDLE',
      scenario: null,
      userGoal: null,
      model: null,
      currentStep: null,
      currentAction: null,
      currentTool: null,
      events: [],
      trajectory: [],
      trajectoryLength: 0,
      untrustedObservation: null,
      injectionNote: null,
      probabilities: null,
      evaluations: [],
      thresholdCrossed: false,
      gateDecision: null,
      gateDetails: null,
      blockedActionReason: null,
      finalAnswer: null,
      toolExecutionCounts: {},
      error: null
    }));
  };

  return {
    state,
    startRun,
    stopRun,
    reset
  };
}
