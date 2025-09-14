from typing import TypedDict

class QueryOptimizer(TypedDict):
    input: str
    output: str
    

class GraphState(TypedDict):
    query: str
    result: str
