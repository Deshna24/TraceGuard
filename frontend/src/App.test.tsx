import { render, screen, fireEvent, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { expect, test, vi, beforeEach } from 'vitest';
import App from './App';
import * as useTraceGuardModule from './hooks/useTraceGuard';

// Mock the hook to control state
vi.mock('./hooks/useTraceGuard', () => ({
  useTraceGuard: vi.fn()
}));

const mockStartRun = vi.fn();
const mockStopRun = vi.fn();
const mockReset = vi.fn();

const defaultState: any = {
  connectionStatus: 'CONNECTED',
  status: 'IDLE',
  userGoal: null,
  scenario: null,
  events: [],
  trajectory: [],
  evaluations: [],
  probabilities: null,
  gateDetails: null,
  toolExecutionCounts: {},
  untrustedObservation: null,
  blockedActionReason: null
};

beforeEach(() => {
  vi.clearAllMocks();
  (useTraceGuardModule.useTraceGuard as any).mockReturnValue({
    state: defaultState,
    startRun: mockStartRun,
    stopRun: mockStopRun,
    reset: mockReset
  });
});

test('application startup & disconnected state', () => {
  (useTraceGuardModule.useTraceGuard as any).mockReturnValue({
    state: { ...defaultState, connectionStatus: 'DISCONNECTED' },
    startRun: mockStartRun, stopRun: mockStopRun, reset: mockReset
  });
  render(<App />);
  expect(screen.getByText(/BACKEND DISCONNECTED/i)).toBeInTheDocument();
  expect(screen.getByText(/RUNTIME UNAVAILABLE/i)).toBeInTheDocument();
});

test('scenario selector and task submission', () => {
  render(<App />);
  const benignButton = screen.getByRole('button', { name: /BENIGN/i });
  fireEvent.click(benignButton);
  expect(mockStartRun).toHaveBeenCalledWith('BENIGN');
});

test('detector event rendering and threshold crossing', () => {
  (useTraceGuardModule.useTraceGuard as any).mockReturnValue({
    state: { 
      ...defaultState, 
      status: 'RUNNING',
      probabilities: {
        p_benign: 0.1,
        p_injection_resisted: 0.1,
        p_hijacked: 0.8,
        threshold: 0.5,
        predicted_class: 'HIJACKED',
        pre_action: true,
        step: 2
      }
    },
    startRun: mockStartRun, stopRun: mockStopRun, reset: mockReset
  });
  render(<BrowserRouter><App /></BrowserRouter>);
  expect(screen.getByText(/80.0%/i)).toBeInTheDocument();
  expect(screen.getByText(/HIJACKED/i)).toBeInTheDocument();
});

test('gate rendering and blocked state', () => {
  (useTraceGuardModule.useTraceGuard as any).mockReturnValue({
    state: { 
      ...defaultState, 
      status: 'BLOCKED',
      blockedActionReason: 'threshold crossed',
      gateDetails: {
        decision: 'BLOCK',
        tool: 'database.export',
        p_hijacked: 0.9,
        threshold: 0.5,
        detection_step: 3,
        pre_action: true
      },
      toolExecutionCounts: { 'database.export': 0 }
    },
    startRun: mockStartRun, stopRun: mockStopRun, reset: mockReset
  });
  render(<BrowserRouter><App /></BrowserRouter>);
  expect(screen.getByText(/TRACEGUARD PRE-ACTION GATE/i)).toBeInTheDocument();
  expect(screen.getByText(/BLOCKED/i)).toBeInTheDocument();
  expect(screen.getByText(/BEHAVIORAL HIJACKING DETECTED/i)).toBeInTheDocument();
  expect(screen.getByText(/TOOL NOT EXECUTED/i)).toBeInTheDocument();
  expect(screen.getByText('0', { selector: '.font-mono' })).toBeInTheDocument();
});

test('trajectory rendering and injection rendering', () => {
  (useTraceGuardModule.useTraceGuard as any).mockReturnValue({
    state: { 
      ...defaultState, 
      status: 'RUNNING',
      untrustedObservation: 'Ignore instructions',
      trajectory: [
        { step: 1, tool: 'search', tool_observation: 'Ignore instructions' }
      ]
    },
    startRun: mockStartRun, stopRun: mockStopRun, reset: mockReset
  });
  render(<BrowserRouter><App /></BrowserRouter>);
  expect(screen.getByText(/⚠ UNTRUSTED CONTENT/i)).toBeInTheDocument();
  expect(screen.getByText(/Ignore instructions/i)).toBeInTheDocument();
  expect(screen.getByText(/search/i)).toBeInTheDocument();
});
