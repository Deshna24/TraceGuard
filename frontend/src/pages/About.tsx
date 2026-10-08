import { ArrowRight, ShieldCheck, ShieldAlert, Cpu } from 'lucide-react';

export const About = () => {
  return (
    <div className="min-h-screen bg-background text-gray-100 p-8">
      <div className="max-w-5xl mx-auto space-y-12">
        
        {/* HEADER */}
        <div>
          <h1 className="text-3xl font-bold tracking-wider text-blue-400 mb-2">ABOUT TRACEGUARD</h1>
          <p className="text-gray-400 text-lg">Real-time LLM agent behavior monitoring and pre-action enforcement.</p>
        </div>

        {/* EXPLANATION */}
        <section className="glass-panel p-8 border border-gray-800 space-y-4">
          <h2 className="text-2xl font-bold text-white mb-4">How It Works</h2>
          <div className="text-gray-300 text-lg leading-relaxed space-y-4">
            <p>
              TRACEGUARD monitors the behavior of an LLM agent as it works through a task.
            </p>
            <p>
              During normal execution, untrusted information from the environment (such as search results or web pages) can attempt to redirect the agent away from its original goal.
            </p>
            <p>
              The system continuously observes changes in the agent's behavior and evaluates whether any proposed action should be allowed. 
            </p>
            <p>
              Crucially, every dangerous action is checked <strong>before execution</strong>. If the action is determined to be unsafe or unauthorized, execution is immediately prevented.
            </p>
          </div>
        </section>

        {/* ARCHITECTURE / SECURITY FLOW */}
        <section className="space-y-6">
          <h2 className="text-2xl font-bold text-blue-400 mb-4 tracking-wider">SYSTEM ARCHITECTURE</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* NORMAL FLOW */}
            <div className="glass-panel p-6 border border-emerald-900/30">
              <h3 className="text-xl font-bold text-emerald-400 mb-6 flex items-center gap-2">
                <ShieldCheck className="w-5 h-5" /> Normal Security Flow
              </h3>
              
              <div className="flex flex-col space-y-3 font-mono text-sm">
                <div className="p-3 bg-gray-900/50 border border-gray-800 rounded text-center text-gray-300">USER GOAL</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-gray-900/50 border border-gray-800 rounded text-center text-gray-300">AGENT</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-gray-900/50 border border-gray-800 rounded text-center text-gray-300">ENVIRONMENT / TOOLS</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-emerald-950/20 border border-emerald-900/50 rounded text-center text-emerald-400 font-bold">UNTRUSTED OBSERVATION</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-gray-900/50 border border-gray-800 rounded text-center text-gray-300">TRAJECTORY</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-blue-950/20 border border-blue-900/50 rounded text-center text-blue-400 font-bold">SECURITY ANALYSIS</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-blue-950/20 border border-blue-900/50 rounded text-center text-blue-400 font-bold">AUTHORIZATION / POLICY</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-blue-950/20 border border-blue-900/50 rounded text-center text-blue-400 font-bold">PRE-ACTION GATE</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-emerald-950/20 border border-emerald-900/50 rounded text-center text-emerald-400 font-bold">ALLOW</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-600 rotate-90" /></div>
                
                <div className="p-3 bg-gray-900/50 border border-gray-800 rounded text-center text-gray-300">TOOL EXECUTION</div>
              </div>
            </div>

            {/* ATTACK FLOW */}
            <div className="glass-panel p-6 border border-red-900/30">
              <h3 className="text-xl font-bold text-red-500 mb-6 flex items-center gap-2">
                <ShieldAlert className="w-5 h-5" /> Attack Interception Path
              </h3>
              
              <div className="flex flex-col space-y-3 font-mono text-sm">
                <div className="p-3 bg-amber-950/20 border border-amber-900/50 rounded text-center text-amber-500 font-bold">UNTRUSTED CONTENT</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-amber-900/50 rotate-90" /></div>
                
                <div className="p-3 bg-amber-950/20 border border-amber-900/50 rounded text-center text-amber-500 font-bold">INJECTION</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-red-900/50 rotate-90" /></div>
                
                <div className="p-3 bg-red-950/20 border border-red-900/50 rounded text-center text-red-400 font-bold">BEHAVIORAL DEVIATION</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-red-900/50 rotate-90" /></div>
                
                <div className="p-3 bg-red-950/20 border border-red-900/50 rounded text-center text-red-400 font-bold">DANGEROUS ACTION</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-blue-900/50 rotate-90" /></div>
                
                <div className="p-3 bg-blue-950/20 border border-blue-900/50 rounded text-center text-blue-400 font-bold">SECURITY ANALYSIS</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-blue-900/50 rotate-90" /></div>

                <div className="p-3 bg-blue-950/20 border border-blue-900/50 rounded text-center text-blue-400 font-bold">PRE-ACTION GATE</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-red-900/50 rotate-90" /></div>
                
                <div className="p-3 bg-red-950/50 border border-red-500/50 rounded text-center text-white font-bold tracking-widest">INTERCEPT / BLOCK</div>
                <div className="flex justify-center"><ArrowRight className="w-4 h-4 text-gray-800 rotate-90" /></div>
                
                <div className="p-3 bg-gray-900 border border-gray-800 rounded text-center text-gray-500 font-bold">TOOL NEVER EXECUTED</div>
              </div>
            </div>
          </div>
        </section>

        {/* LIMITATIONS / RESEARCH */}
        <section className="glass-panel p-6 border border-gray-800 bg-gray-900/30">
          <h2 className="text-xl font-bold text-gray-300 mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5" /> Research Context & Limitations
          </h2>
          <div className="text-gray-400 space-y-4 text-sm leading-relaxed">
            <p>
              TRACEGUARD is a research prototype demonstrating a trajectory-level approach to LLM agent security. The security model conceptually combines several signals: authorized goal, trust/provenance, agent action, tool privilege, and goal-action alignment.
            </p>
            <p>
              The runtime demonstrations provided in this interface are illustrative deployments of a frozen trajectory evaluation system. TRACEGUARD is not intended to claim universal real-world protection against all possible attacks. Its effectiveness depends heavily on the quality of its evaluation components and the specific configuration of the agent environment.
            </p>
          </div>
        </section>

      </div>
    </div>
  );
};
