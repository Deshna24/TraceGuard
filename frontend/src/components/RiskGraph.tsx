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
  const data = evaluations.map((e, idx) => ({
    step: `Step ${idx + 1}`,
    pHijacked: Number(e.p_hijacked || 0),
    baseline: Number(e.baseline_p_hijacked || 0),
    predictedClass: e.predicted_class,
  }));

  if (data.length === 0) {
    return null;
  }

  return (
    <div className="glass-panel p-4 flex flex-col gap-2" style={{ backgroundColor: '#111827', borderColor: '#1F2937', borderRadius: '0.5rem', borderStyle: 'solid', borderWidth: '1px' }}>
      <div className="flex justify-between items-center border-b border-gray-800 pb-2">
        <h2 className="text-sm font-bold flex items-center gap-2 text-gray-400">
          <Activity className="w-4 h-4 text-blue-400" />
          SECURITY RISK TRAJECTORY
        </h2>
        <span className="text-[10px] text-gray-500 font-mono">P(HIJACKED) vs STEP</span>
      </div>
      <div className="h-[180px] w-full mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 10, right: 15, left: -20, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
            <XAxis dataKey="step" stroke="#6B7280" tick={{ fontSize: 10 }} />
            <YAxis stroke="#6B7280" domain={[0, 1]} tick={{ fontSize: 10 }} ticks={[0, 0.25, 0.5, 0.75, 1]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', color: '#F3F4F6', fontSize: '12px' }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: any, name?: any) => [
                `${(Number(value) * 100).toFixed(1)}% (${Number(value).toFixed(3)})`,
                name === 'pHijacked' ? 'P(HIJACKED)' : 'Baseline Risk'
              ]}
            />
            <ReferenceLine 
              y={0.5} 
              stroke="#EF4444" 
              strokeDasharray="4 4" 
              strokeWidth={1.5}
              label={{ position: 'insideTopLeft', value: 'BLOCK THRESHOLD (0.50)', fill: '#EF4444', fontSize: 10 }} 
            />
            
            <Line
              type="monotone"
              dataKey="pHijacked"
              name="pHijacked"
              stroke="#3B82F6"
              strokeWidth={2.5}
              isAnimationActive={false}
              dot={(props) => {
                const { cx, cy, payload } = props;
                if (payload.pHijacked >= 0.5) {
                  return (
                    <circle key={`dot-${cx}-${cy}`} cx={cx} cy={cy} r={6} fill="#EF4444" stroke="#FEE2E2" strokeWidth={2} />
                  );
                }
                return <circle key={`dot-${cx}-${cy}`} cx={cx} cy={cy} r={4} fill="#3B82F6" stroke="#1E3A8A" strokeWidth={1} />;
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
