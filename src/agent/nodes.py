import json
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import yaml

from .state import AgentState
from .tools import GET_TOOLS

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Initialize LLM
llm = ChatOllama(
    model=config['llm']['model_name'],
    temperature=config['llm']['temperature']
)

# Bind tools
llm_with_tools = llm.bind_tools(GET_TOOLS)

def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # We create a step record for the logger
    step_record = {
        "action": "llm_decision",
        "content": response.content,
        "tool_calls": [tc["name"] for tc in response.tool_calls] if hasattr(response, "tool_calls") and response.tool_calls else []
    }
    state["trajectory_steps"].append(step_record)
    
    return {"messages": [response], "trajectory_steps": state["trajectory_steps"]}

def tool_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # We will simulate tool execution here or in the graph
    # For now, let's just create tool messages
    tool_messages = []
    
    tools_by_name = {t.name: t for t in GET_TOOLS}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Here we would normally call the tool
        # For traceguard, the simulator might intercept this output before it goes to the agent
        # We will handle interception in the main generator loop by mutating state or replacing the node
        
        tool_func = tools_by_name.get(tool_name)
        if tool_func:
            try:
                # Call tool
                result = tool_func.invoke(tool_args)
            except Exception as e:
                result = f"Error: {e}"
        else:
            result = f"Tool {tool_name} not found."
            
        tool_message = ToolMessage(
            content=str(result),
            name=tool_name,
            tool_call_id=tool_call["id"]
        )
        tool_messages.append(tool_message)
        
        step_record = {
            "action": "tool_call",
            "tool": tool_name,
            "tool_input": tool_args,
            "observation": str(result)
        }
        state["trajectory_steps"].append(step_record)
        
    return {"messages": tool_messages, "trajectory_steps": state["trajectory_steps"]}

def route_after_agent(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END
