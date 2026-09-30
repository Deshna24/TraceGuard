from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes import agent_node, tool_node, route_after_agent

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# Global graph instance
agent_graph = build_graph()
