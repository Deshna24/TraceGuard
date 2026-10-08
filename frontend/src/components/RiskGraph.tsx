import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer
} from 'recharts';
import { DetectorEvaluation } from '../types';

interface RiskGraphProps {
  evaluations: DetectorEvaluation[];
}

export const RiskGraph: React.FC<RiskGraphProps> = ({ evaluations }) => {
  const data = evaluations.map((ev) => ({
    step: `Step ${ev.trajectory_step}`,
    pHijacked: ev.P_HIJACKED,
  }));

  return (
    <div className="w-full h-64 glass-panel p-4">
      <h3 className="text-lg font-semibold mb-4 text-gray-200">Live Risk Graph</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="step" stroke="#9CA3AF" />
          <YAxis domain={[0, 1]} stroke="#9CA3AF" />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '0.375rem' }}
            itemStyle={{ color: '#F3F4F6' }}
          />
          <ReferenceLine y={0.5} stroke="#EF4444" strokeDasharray="3 3" label={{ position: 'top', value: 'Threshold 0.5', fill: '#EF4444' }} />
          <Line 
            type="monotone" 
            dataKey="pHijacked" 
            stroke="#F59E0B" 
            strokeWidth={3}
            dot={{ fill: '#F59E0B', r: 6 }}
            activeDot={{ r: 8 }}
            isAnimationActive={true}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
