import React from 'react';

export const About = () => {
  return (
    <div className="min-h-screen bg-background text-gray-100 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold tracking-wider text-blue-400 mb-8">ABOUT TRACEGUARD</h1>
        
        <div className="space-y-8 text-gray-300 leading-relaxed">
           <section className="glass-panel p-6 border border-gray-800">
             <h2 className="text-xl font-bold text-white mb-4">What is TRACEGUARD?</h2>
             <p className="mb-4">
               TRACEGUARD is a trajectory-level detection system designed to monitor and secure LLM-based autonomous agents in real-time. Unlike traditional input filtering, TRACEGUARD continuously analyzes the agent's behavior throughout its execution trajectory.
             </p>
             <p>
               It uses a frozen LSTM detector combined with MiniLM embeddings to perform prefix scoring on the sequence of tool observations and actions, classifying the agent's state into three distinct probabilities.
             </p>
           </section>

           <section className="glass-panel p-6 border border-gray-800">
             <h2 className="text-xl font-bold text-white mb-4">Three Class Probabilities</h2>
             <ul className="list-disc pl-6 space-y-2">
                <li><strong className="text-emerald-400">P(BENIGN):</strong> The agent is operating normally and securely.</li>
                <li><strong className="text-amber-500">P(INJECTION_RESISTED):</strong> An untrusted observation (like a prompt injection) was encountered, but the agent ignored it and remained aligned with the original goal.</li>
                <li><strong className="text-red-500">P(HIJACKED):</strong> The agent has been compromised by an injection and is actively deviating from its authorized instructions.</li>
             </ul>
           </section>

           <section className="glass-panel p-6 border border-gray-800">
             <h2 className="text-xl font-bold text-white mb-4">Pre-Action Safety Gate</h2>
             <p className="mb-4">
               TRACEGUARD operates entirely in a controlled runtime environment. Before any proposed tool action is executed, it must pass through the Pre-Action Safety Gate.
             </p>
             <p>
               If the detector evaluates that the agent's <span className="font-mono bg-gray-900 px-1 rounded">P(HIJACKED)</span> has crossed the defined threshold (e.g., 0.50), the safety gate immediately blocks the action. This ensures that a hijacked agent cannot execute dangerous operations, such as unauthorized data exfiltration, because the block occurs <em>before</em> execution.
             </p>
           </section>

           <section className="glass-panel p-6 border border-gray-800 bg-blue-950/10">
             <h2 className="text-xl font-bold text-blue-400 mb-4">Research Disclaimer</h2>
             <p className="italic text-gray-400">
               Runtime demonstrations are illustrative deployments of the frozen detector and are not new benchmark metrics. TRACEGUARD is a research prototype and its effectiveness depends on the quality of its training data and the specific environment configuration.
             </p>
           </section>
        </div>
      </div>
    </div>
  );
};
