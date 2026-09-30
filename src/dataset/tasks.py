import random

TASK_FAMILIES = {
    "information_retrieval": [
        {"goal": "Find the population of France.", "expected_tool": "search_knowledge", "query": "population of France"},
        {"goal": "Find the capital of Japan.", "expected_tool": "search_knowledge", "query": "capital of Japan"}
    ],
    "document_tasks": [
        {"goal": "Read document doc_001 and summarize the financial profit.", "expected_tool": "read_document", "doc_id": "doc_001"},
        {"goal": "Check the working hours from document doc_002.", "expected_tool": "read_document", "doc_id": "doc_002"}
    ],
    "multi_step_calculation": [
        {"goal": "Calculate 15 * 8.", "expected_tool": "calculator", "expression": "15 * 8"},
        {"goal": "Evaluate (100 / 4) + 12.", "expected_tool": "calculator", "expression": "(100 / 4) + 12"}
    ],
    "structured_lookup": [
        {"goal": "Lookup the role for user_123.", "expected_tool": "lookup_record", "record_id": "user_123"},
        {"goal": "Find the access level of user_456.", "expected_tool": "lookup_record", "record_id": "user_456"}
    ]
}

def generate_random_task():
    family = random.choice(list(TASK_FAMILIES.keys()))
    task = random.choice(TASK_FAMILIES[family])
    return family, task
