import ReactFlow, { Background, Controls, Edge, Node, MarkerType } from 'reactflow';
import 'reactflow/dist/style.css';

const nodes: Node[] = [
  { id: '1', position: { x: 250, y: 0 }, data: { label: 'USER' }, style: { backgroundColor: '#1e3a8a', color: 'white', fontWeight: 'bold' } },
  { id: '2', position: { x: 250, y: 100 }, data: { label: 'GRANITE AGENT' }, style: { backgroundColor: '#1e40af', color: 'white' } },
  { id: '3', position: { x: 250, y: 200 }, data: { label: 'TRAJECTORY' } },
  { id: '4', position: { x: 250, y: 300 }, data: { label: 'MINILM' } },
  { id: '5', position: { x: 250, y: 400 }, data: { label: 'TRACEGUARD LSTM' }, style: { backgroundColor: '#3730a3', color: 'white' } },
  { id: '6', position: { x: 250, y: 500 }, data: { label: '3 CLASS PROBABILITIES' } },
  { id: '7', position: { x: 250, y: 600 }, data: { label: 'PRE-ACTION GATE' }, style: { backgroundColor: '#111827', color: 'white', border: '1px solid #10b981' } },
  { id: '8', position: { x: 250, y: 700 }, data: { label: 'ALLOW / BLOCK' } },
  { id: '9', position: { x: 250, y: 800 }, data: { label: 'CONTROLLED TOOLS' } },
  
  { id: '10', position: { x: 600, y: 100 }, data: { label: 'TOOL OBSERVATION' } },
  { id: '11', position: { x: 600, y: 200 }, data: { label: 'UNTRUSTED CONTENT' }, style: { border: '1px solid #f59e0b' } },
  { id: '12', position: { x: 600, y: 300 }, data: { label: 'INJECTION' }, style: { border: '1px solid #f59e0b', backgroundColor: '#78350f', color: 'white' } },
  { id: '13', position: { x: 600, y: 400 }, data: { label: 'BEHAVIORAL DRIFT' }, style: { border: '1px solid #ef4444', backgroundColor: '#7f1d1d', color: 'white' } },
];

const edges: Edge[] = [
  { id: 'e1-2', source: '1', target: '2', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e2-3', source: '2', target: '3', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e3-4', source: '3', target: '4', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e4-5', source: '4', target: '5', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e5-6', source: '5', target: '6', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e6-7', source: '6', target: '7', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e7-8', source: '7', target: '8', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e8-9', source: '8', target: '9', markerEnd: { type: MarkerType.ArrowClosed } },
  
  { id: 'e10-11', source: '10', target: '11', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e11-12', source: '11', target: '12', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e12-13', source: '12', target: '13', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e13-2', source: '13', target: '2', animated: true, markerEnd: { type: MarkerType.ArrowClosed }, style: { stroke: '#ef4444' } },
  
  { id: 'e9-10', source: '9', target: '10', type: 'step', markerEnd: { type: MarkerType.ArrowClosed } },
];

export const Architecture = () => {
  return (
    <div className="h-screen w-full bg-background flex flex-col">
      <div className="p-6">
        <h1 className="text-3xl font-bold tracking-wider text-blue-400">ARCHITECTURE</h1>
        <p className="text-gray-400 text-sm mt-1">TRACEGUARD Flow Diagram</p>
      </div>
      <div className="flex-1">
        <ReactFlow 
          nodes={nodes} 
          edges={edges} 
          fitView 
          className="bg-gray-900"
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
        >
          <Background color="#374151" gap={16} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
};
