import React, { useEffect, useState } from 'react';
import { CheckCircle2, XCircle, RefreshCcw, AlertCircle } from 'lucide-react';

interface ModelMeta {
  id: string;
  name: string;
  architecture: string;
  parameters: number;
  checkpoint_path: string;
  checkpoint_available: boolean;
  supported_outputs: string[];
  research_track: string;
  macro_f1: number;
  hijacked_f1: number;
  description: string;
  is_loaded: boolean;
  device: string;
}

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchModels = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch('http://localhost:8000/api/models');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: ModelMeta[] = await res.json();
      setModels(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to fetch model registry.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  return (
    <div className="min-h-screen bg-background text-gray-100 p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-wider text-blue-400">MODEL REGISTRY & CONFIGURATION</h1>
          <p className="text-gray-400 text-sm mt-1">
            Registered temporal deep learning architectures for prefix evaluation, parameter allocations, and hardware execution status.
          </p>
        </div>
        <button
          onClick={fetchModels}
          className="px-3 py-1.5 bg-gray-900 hover:bg-gray-800 border border-gray-700 text-xs font-mono rounded text-gray-300 flex items-center gap-2 transition-colors"
        >
          <RefreshCcw className="w-3.5 h-3.5" /> Refresh Status
        </button>
      </div>

      {isLoading ? (
        <div className="glass-panel p-12 text-center text-gray-400 font-mono text-xs flex items-center justify-center gap-2">
          <RefreshCcw className="w-4 h-4 animate-spin text-blue-400" /> Loading model registry...
        </div>
      ) : errorMsg ? (
        <div className="p-4 bg-red-950/40 border border-red-700 rounded text-red-300 text-xs flex items-center gap-2 font-mono">
          <AlertCircle className="w-4 h-4 text-red-400" /> {errorMsg}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {models.map((m) => (
            <div
              key={m.id}
              className="glass-panel p-6 rounded-lg border border-gray-800 bg-gray-950/40 flex flex-col justify-between space-y-4"
            >
              <div className="space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-blue-900/40 text-blue-300 border border-blue-800">
                      {m.research_track}
                    </span>
                    <h2 className="text-lg font-bold text-gray-200 mt-2 font-mono">{m.name}</h2>
                  </div>
                  <div className="flex items-center gap-1.5 font-mono text-xs">
                    {m.checkpoint_available ? (
                      <span className="flex items-center gap-1 text-emerald-400 bg-emerald-950/40 border border-emerald-800 px-2 py-0.5 rounded">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-red-400 bg-red-950/40 border border-red-800 px-2 py-0.5 rounded">
                        <XCircle className="w-3.5 h-3.5" /> Missing
                      </span>
                    )}
                  </div>
                </div>

                <p className="text-xs text-gray-400 leading-relaxed">{m.description}</p>

                <div className="p-3 bg-gray-900/60 rounded border border-gray-800 space-y-2 text-xs font-mono">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Architecture:</span>
                    <span className="text-gray-300 text-right text-[11px] max-w-[280px]">{m.architecture}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Parameter Count:</span>
                    <span className="text-blue-300 font-bold">{m.parameters.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Execution Hardware:</span>
                    <span className="text-gray-300">{m.device.toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Benchmark Macro-F1:</span>
                    <span className="text-emerald-400 font-bold">{m.macro_f1.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Hijacked-Class F1:</span>
                    <span className="text-emerald-400 font-bold">{m.hijacked_f1.toFixed(4)}</span>
                  </div>
                </div>
              </div>

              <div>
                <span className="text-[11px] font-mono text-gray-500 uppercase font-bold block mb-1.5">
                  Supported Model Outputs:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {m.supported_outputs.map((op, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 bg-gray-900 text-gray-300 border border-gray-700 rounded text-[10px] font-mono"
                    >
                      {op}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
