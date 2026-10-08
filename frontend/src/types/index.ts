export interface TraceGuardEvent {
  event_type: string;
  data: any;
}

export interface GateDecision {
  proposed_action: any;
  decision: 'ALLOW' | 'BLOCK';
  p_hijacked: number;
  threshold: number;
  detection_step: number;
  pre_action: boolean;
}

export interface AgentAction {
  action: string;
  tool: string;
  tool_input: any;
}

export interface TrajectoryStep {
  step: number;
  action: string;
  tool: string;
  tool_input: any;
  tool_observation: any;
  state: string;
}

export interface DetectorEvaluation {
  trajectory_step: number;
  P_BENIGN: number;
  P_INJECTION_RESISTED: number;
  P_HIJACKED: number;
  predicted_class: string;
  threshold: number;
  pre_action: boolean;
}
