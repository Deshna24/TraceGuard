import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { Activity } from 'lucide-react';

interface RiskGraphProps {
  evaluations: any[];
}

export const RiskGraph: React.FC<RiskGraphProps> = ({ evaluations }) => {
  const data = evaluations.map((e) => ({
    step: e.step,
    pHijacked: e.p_hijacked,
  }));

  const thresholdCrossedIndex = data.findIndex((d) => d.pHijacked >= 0.5);
  
  return (
    <div className="glass-panel p-6 flex flex-col gap-4" style={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '0.5rem', borderStyle: 'solid', borderWidth: '1px' }}>
      <h2 className="text-xl font-bold border-b border-gray-800 pb-2 flex items-center gap-2">
        <Activity className="w-5 h-5 text-blue-400" />
        RISK GRAPH
      </h2>
      <div className="h-[250px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
            <XAxis dataKey="step" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" domain={[0, 1]} tickFormatter={(val) => val.toFixed(1)} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', color: '#F3F4F6' }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: any) => [Number(value).toFixed(3), 'P(HIJACKED)']}
            />
            <ReferenceLine y={0.5} stroke="#EF4444" strokeDasharray="3 3" label={{ position: 'top', value: '0.50 THRESHOLD', fill: '#EF4444', fontSize: 12 }} />
            
            <Line
              type="monotone"
              dataKey="pHijacked"
              stroke="#3B82F6"
              strokeWidth={3}
              dot={(props) => {
                const { cx, cy, payload } = props;
                if (payload.pHijacked >= 0.5) {
                  return (
                    <circle cx={cx} cy={cy} r={6} fill="#EF4444" stroke="#7F1D1D" strokeWidth={2} />
                  );
                }
                return <circle cx={cx} cy={cy} r={4} fill="#3B82F6" stroke="none" />;
              }}
              activeDot={{ r: 8 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {thresholdCrossedIndex !== -1 && (
         <div className="text-center font-bold text-red-500 bg-red-950/30 py-2 border border-red-900/50 rounded animate-pulse mt-2">
            🚨 THRESHOLD CROSSED AT STEP {data[thresholdCrossedIndex].step}
         </div>
      )}
    </div>
  );
};
