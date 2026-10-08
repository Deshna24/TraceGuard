export interface EventEnvelope {
  run_id: string;
  event_id: string;
  timestamp: string;
  event_type: string;
  step: number | null;
  payload: any;
}

export type EventType =
  | 'RUN_STARTED'
  | 'USER_TASK_RECEIVED'
  | 'AGENT_STARTED'
  | 'RUN_COMPLETED'
  | 'RUNTIME_ERROR'
  | 'ACTION_PROPOSED'
  | 'TOOL_STARTED'
  | 'TOOL_COMPLETED'
  | 'OBSERVATION_RECEIVED'
  | 'TRAJECTORY_UPDATED'
  | 'INJECTION_OBSERVED'
  | 'DETECTOR_EVALUATED'
  | 'THRESHOLD_CROSSED'
  | 'GATE_DECISION'
  | 'ACTION_BLOCKED';

export interface TrajectoryStep {
  step: number;
  action: string;
  tool: string;
  tool_input: any;
  tool_observation: any;
  state: string;
}

export interface RunState {
  runId: string | null;
  status: 'IDLE' | 'RUNNING' | 'ACTING' | 'WAITING' | 'COMPLETED' | 'BLOCKED' | 'ERROR';
  scenario: string | null;
  userGoal: string | null;
  model: string | null;
  currentStep: number | null;
  currentAction: any | null;
  currentTool: string | null;
  events: EventEnvelope[];
  trajectory: TrajectoryStep[];
  trajectoryLength: number;
  untrustedObservation: string | null;
  injectionNote: string | null;
  probabilities: {
    p_benign: number;
    p_injection_resisted: number;
    p_hijacked: number;
    threshold: number;
    predicted_class: string;
    pre_action: boolean;
    step: number;
  } | null;
  evaluations: any[]; // for RiskGraph
  thresholdCrossed: boolean;
  gateDecision: 'ALLOW' | 'BLOCK' | null;
  gateDetails: any | null;
  blockedActionReason: string | null;
  finalAnswer: string | null;
  toolExecutionCounts: Record<string, number>;
  error: string | null;
  connectionStatus: 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED';
}
