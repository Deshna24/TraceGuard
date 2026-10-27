import React, { useEffect, useState } from 'react';
import { Cpu, HardDrive, CheckCircle2, XCircle, RefreshCcw, Shield, Layers } from 'lucide-react';

interface SystemHealthData {
  status: string;
  version: string;
  device: string;
  device_name: string;
  cuda_available: boolean;
  vram_total_gb?: number;
  vram_allocated_gb?: number;
  torch_version: string;
  embedding_cache_available: boolean;
  models_available: Record<string, boolean>;
}

export const SystemHealthPage: React.FC = () => {
  const [health, setHealth] = useState<SystemHealthData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchHealth = async () => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch('http://localhost:8000/api/system/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: SystemHealthData = await res.json();
      setHealth(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to fetch system health.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="min-h-screen bg-background text-gray-100 p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-wider text-blue-400">SYSTEM HEALTH & HARDWARE DIAGNOSTICS</h1>
          <p className="text-gray-400 text-sm mt-1">
            Local execution device, CUDA acceleration status, VRAM consumption, offline model checkpoints, and runtime readiness.
          </p>
        </div>
        <button
          onClick={fetchHealth}
          className="px-3 py-1.5 bg-gray-900 hover:bg-gray-800 border border-gray-700 text-xs font-mono rounded text-gray-300 flex items-center gap-2 transition-colors"
        >
          <RefreshCcw className="w-3.5 h-3.5" /> Refresh Diagnostics
        </button>
      </div>

      {isLoading ? (
        <div className="glass-panel p-12 text-center text-gray-400 font-mono text-xs flex items-center justify-center gap-2">
          <RefreshCcw className="w-4 h-4 animate-spin text-blue-400" /> Querying system diagnostics...
        </div>
      ) : errorMsg ? (
        <div className="p-4 bg-red-950/40 border border-red-700 rounded text-red-300 text-xs flex items-center gap-2 font-mono">
          <XCircle className="w-4 h-4 text-red-400" /> {errorMsg}
        </div>
      ) : health ? (
        <div className="space-y-6">
          {/* Top Status Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="glass-panel p-4 rounded-lg border border-gray-800 bg-gray-950/40">
              <span className="text-[11px] font-mono text-gray-500 uppercase font-bold block">Backend Service</span>
              <div className="flex items-center gap-1.5 mt-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span className="text-lg font-bold text-gray-200 font-mono capitalize">{health.status}</span>
              </div>
              <span className="text-[10px] text-gray-500 font-mono mt-1 block">API Version: {health.version}</span>
            </div>

            <div className="glass-panel p-4 rounded-lg border border-gray-800 bg-gray-950/40">
              <span className="text-[11px] font-mono text-gray-500 uppercase font-bold block">Hardware Acceleration</span>
              <div className="flex items-center gap-1.5 mt-2">
                <Cpu className="w-4 h-4 text-blue-400" />
                <span className="text-lg font-bold text-blue-300 font-mono">{health.device.toUpperCase()}</span>
              </div>
              <span className="text-[10px] text-gray-400 font-mono mt-1 block truncate">{health.device_name}</span>
            </div>

            <div className="glass-panel p-4 rounded-lg border border-gray-800 bg-gray-950/40">
              <span className="text-[11px] font-mono text-gray-500 uppercase font-bold block">GPU Memory (VRAM)</span>
              <div className="flex items-center gap-1.5 mt-2">
                <HardDrive className="w-4 h-4 text-purple-400" />
                <span className="text-lg font-bold text-purple-300 font-mono">
                  {health.vram_total_gb ? `${health.vram_total_gb} GB` : 'N/A'}
                </span>
              </div>
              <span className="text-[10px] text-gray-500 font-mono mt-1 block">
                Allocated: {health.vram_allocated_gb !== undefined ? `${health.vram_allocated_gb} GB` : '0 GB'}
              </span>
            </div>

            <div className="glass-panel p-4 rounded-lg border border-gray-800 bg-gray-950/40">
              <span className="text-[11px] font-mono text-gray-500 uppercase font-bold block">Embedding Model Cache</span>
              <div className="flex items-center gap-1.5 mt-2">
                {health.embedding_cache_available ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span className="text-lg font-bold text-emerald-400 font-mono">Cached</span>
                  </>
                ) : (
                  <>
                    <XCircle className="w-4 h-4 text-amber-400" />
                    <span className="text-lg font-bold text-amber-400 font-mono">Online</span>
                  </>
                )}
              </div>
              <span className="text-[10px] text-gray-500 font-mono mt-1 block">all-MiniLM-L6-v2</span>
            </div>
          </div>

          {/* Model Checkpoint Readiness Matrix */}
          <div className="glass-panel p-6 rounded-lg border border-gray-800 bg-gray-950/50 space-y-4">
            <h2 className="text-sm font-bold tracking-wider text-gray-200 uppercase font-mono flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-400" /> Neural Checkpoint Availability
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
              {Object.entries(health.models_available).map(([modelId, isAvailable]) => (
                <div
                  key={modelId}
                  className="p-3 bg-gray-900 border border-gray-800 rounded flex items-center justify-between"
                >
                  <span className="uppercase text-gray-300 font-bold">{modelId}</span>
                  {isAvailable ? (
                    <span className="flex items-center gap-1 text-emerald-400 text-[11px]">
                      <CheckCircle2 className="w-3.5 h-3.5" /> AVAILABLE
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-400 text-[11px]">
                      <XCircle className="w-3.5 h-3.5" /> MISSING
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Runtime Environment Info */}
          <div className="glass-panel p-6 rounded-lg border border-gray-800 bg-gray-950/50 space-y-3 font-mono text-xs">
            <h2 className="text-sm font-bold tracking-wider text-gray-200 uppercase flex items-center gap-2">
              <Shield className="w-4 h-4 text-blue-400" /> Environment Configuration
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-gray-400">
              <div>
                <span className="text-gray-500 block">PyTorch Framework Version:</span>
                <span className="text-gray-200 font-bold">{health.torch_version}</span>
              </div>
              <div>
                <span className="text-gray-500 block">Execution Mode:</span>
                <span className="text-gray-200 font-bold">
                  {health.cuda_available ? 'Hardware GPU Accelerated (CUDA)' : 'Host CPU Fallback'}
                </span>
              </div>
              <div>
                <span className="text-gray-500 block">Local REST Service:</span>
                <span className="text-gray-200 font-bold">http://127.0.0.1:8000</span>
              </div>
              <div>
                <span className="text-gray-500 block">Local WebSocket Channel:</span>
                <span className="text-gray-200 font-bold">ws://127.0.0.1:8000/ws</span>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
