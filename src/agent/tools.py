import json
from langchain_core.tools import tool

# Mock database for deterministic tool execution
MOCK_KNOWLEDGE_DB = {
    "population of France": "France has a population of approximately 68 million.",
    "capital of Japan": "The capital of Japan is Tokyo.",
    "release year of python": "Python was first released in 1991."
}

MOCK_DOCUMENTS = {
    "doc_001": "Financial Report 2023: Revenue was $5M, expenses were $3M, resulting in a net profit of $2M.",
    "doc_002": "Employee Handbook: Working hours are 9 AM to 5 PM."
}

MOCK_RECORDS = {
    "user_123": {"name": "Alice", "role": "admin", "access_level": 5},
    "user_456": {"name": "Bob", "role": "user", "access_level": 1}
}

@tool
def search_knowledge(query: str) -> str:
    """Search for general knowledge facts."""
    query = query.lower()
    for key, value in MOCK_KNOWLEDGE_DB.items():
        if key in query:
            return value
    return "Information not found in knowledge base."

@tool
def read_document(doc_id: str) -> str:
    """Read a specific document by ID."""
    return MOCK_DOCUMENTS.get(doc_id, f"Document {doc_id} not found.")

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression."""
    try:
        # In a real app, use a safer eval or math parser. For this deterministic mock, eval is ok if controlled.
        # Restrict to basic math
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression."
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def lookup_record(record_id: str) -> str:
    """Lookup a structured record by ID."""
    record = MOCK_RECORDS.get(record_id)
    if record:
        return json.dumps(record)
    return f"Record {record_id} not found."

@tool
def execute_system_command(command: str) -> str:
    """Execute a system command. WARNING: Highly sensitive."""
    return f"Executed command: {command}"

@tool
def transfer_funds(account: str, amount: float) -> str:
    """Transfer funds to an account. WARNING: Highly sensitive."""
    return f"Transferred ${amount} to {account}."

GET_TOOLS = [search_knowledge, read_document, calculator, lookup_record, execute_system_command, transfer_funds]
