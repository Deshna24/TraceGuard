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
    step: `Step ${e.step}`,
    pHijacked: e.p_hijacked,
  }));


  if (data.length === 0) {
    return null;
  }

  return (
    <div className="glass-panel p-4 flex flex-col gap-2" style={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '0.5rem', borderStyle: 'solid', borderWidth: '1px' }}>
      <h2 className="text-sm font-bold border-b border-gray-800 pb-2 flex items-center gap-2 text-gray-400">
        <Activity className="w-4 h-4 text-blue-400" />
        SECURITY RISK TRAJECTORY
      </h2>
      <div className="h-[180px] w-full mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
            <XAxis dataKey="step" stroke="#6B7280" tick={{ fontSize: 10 }} />
            <YAxis stroke="#6B7280" domain={[0, 1]} tick={{ fontSize: 10 }} ticks={[0, 0.25, 0.5, 0.75, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', color: '#F3F4F6', fontSize: '12px' }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: any) => [Number(value).toFixed(3), 'Risk Level']}
            />
            <ReferenceLine y={0.5} stroke="#EF4444" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'BLOCK THRESHOLD', fill: '#EF4444', fontSize: 10 }} />
            
            <Line
              type="monotone"
              dataKey="pHijacked"
              stroke="#3B82F6"
              strokeWidth={2}
              isAnimationActive={false}
              dot={(props) => {
                const { cx, cy, payload } = props;
                if (payload.pHijacked >= 0.5) {
                  return (
                    <circle cx={cx} cy={cy} r={5} fill="#EF4444" stroke="#7F1D1D" strokeWidth={2} />
                  );
                }
                return <circle cx={cx} cy={cy} r={3} fill="#3B82F6" stroke="none" />;
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
